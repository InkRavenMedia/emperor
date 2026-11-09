import { useMemo } from "react";
import { useQueries, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { type MonitorData } from "@/lib/api/monitor-data";
import { hasHistoricalMonitorData, hasMonitorData } from "@/lib/monitor/grouping";
import { monitorQueryKeys } from "@/lib/query-keys";
import { createLazyFunction } from "@/lib/lazy-value";
import {
  type HistoricalMonitorRunData,
  type MonitorChartsSource,
} from "@/types/monitor";

const RUNNING_STATUSES = new Set(["running", "queued"]);
const RUNNING_MONITOR_REFETCH_INTERVAL_MS = 1_500;
const HISTORICAL_MONITOR_REQUEST_CONCURRENCY = 2;
type MonitorDataApi = typeof import("@/lib/api/monitor-data");
const fetchLogRunMonitorData: MonitorDataApi["fetchLogRunMonitorData"] =
  createLazyFunction(() =>
    import("@/lib/api/monitor-data").then(
      (module) => module.fetchLogRunMonitorData,
    ),
  );
const fetchMonitorData: MonitorDataApi["fetchMonitorData"] =
  createLazyFunction(() =>
    import("@/lib/api/monitor-data").then((module) => module.fetchMonitorData),
  );

type QueryProgress = {
  loaded: number;
  failed: number;
  total: number;
  isLoading: boolean;
};

type MonitorRequest = Pick<
  UseQueryResult<MonitorData>,
  "isSuccess" | "isError" | "isFetching" | "isLoading" | "refetch"
>;

export type MonitorRequestFailure = {
  failed: number;
  total: number;
  isRetrying: boolean;
  retry: () => void;
};

function monitorRequestFailure(
  requests: MonitorRequest[],
): MonitorRequestFailure | null {
  const failedRequests = requests.filter((request) => request.isError);
  if (failedRequests.length === 0) {
    return null;
  }
  return {
    failed: failedRequests.length,
    total: requests.length,
    isRetrying: failedRequests.some((request) => request.isFetching),
    retry: () => {
      for (const request of failedRequests) {
        if (!request.isFetching) {
          void request.refetch();
        }
      }
    },
  };
}

function createConcurrencyLimiter(limit: number, scopeKey: string) {
  void scopeKey;
  const pending: Array<() => void> = [];
  let active = 0;

  const drain = () => {
    while (active < limit) {
      const start = pending.shift();
      if (!start) {
        return;
      }
      active += 1;
      start();
    }
  };

  return function run<T>(operation: () => Promise<T>) {
    return new Promise<T>((resolve, reject) => {
      pending.push(() => {
        void operation()
          .then(resolve, reject)
          .finally(() => {
            active -= 1;
            drain();
          });
      });
      drain();
    });
  };
}

function queryProgress({
  queries,
  total,
}: {
  queries: MonitorRequest[];
  total: number;
}): QueryProgress {
  const loaded = queries.filter((query) => query.isSuccess).length;
  const failed = queries.filter((query) => query.isError).length;
  const settled = loaded + failed;
  return {
    loaded,
    failed,
    total,
    isLoading:
      total > 0 &&
      settled < total &&
      queries.some((query) => query.isFetching || query.isLoading),
  };
}

function dataWasTruncated(data: MonitorData | undefined) {
  return Boolean(
    data?.truncated ||
      data?.scalarSeries.some((series) => series.truncated) ||
      data?.histograms.some((histogram) => histogram.truncated) ||
      data?.images.some((image) => image.truncated),
  );
}

function historicalDataWasTruncated(results: HistoricalMonitorRunData[] | undefined) {
  return Boolean(results?.some((result) => dataWasTruncated(result.data)));
}

function truncationReason(data: MonitorData | undefined) {
  return (
    data?.truncationReason ??
    data?.scalarSeries.find((series) => series.truncationReason)?.truncationReason ??
    data?.histograms.find((histogram) => histogram.truncationReason)?.truncationReason ??
    data?.images.find((image) => image.truncationReason)?.truncationReason ??
    null
  );
}

function historicalTruncationReason(
  results: HistoricalMonitorRunData[] | undefined,
) {
  return (
    results
      ?.map((result) => truncationReason(result.data))
      .find((reason): reason is string => Boolean(reason)) ?? null
  );
}

type UseMonitorChartQueriesInput = {
  source: MonitorChartsSource;
  nodePath: string;
  dataset: string;
  preset: string;
  comparisonNodePath?: string;
  enabled?: boolean;
};

export function useMonitorChartQueries({
  source,
  nodePath,
  dataset,
  preset,
  comparisonNodePath,
  enabled = true,
}: UseMonitorChartQueriesInput) {
  const activeJob = source.kind === "active-job" ? source.job : undefined;
  const historicalRun = source.kind === "historical-run" ? source.run : undefined;
  const historicalRunGroup =
    source.kind === "historical-run-group" ? source : undefined;
  const historicalRuns = useMemo(
    () => historicalRunGroup?.runs ?? (historicalRun ? [historicalRun] : []),
    [historicalRun, historicalRunGroup?.runs],
  );
  const historicalRunIds = useMemo(
    () => historicalRuns.map((run) => run.id),
    [historicalRuns],
  );
  const historicalRunQueryKey = useMemo(
    () => `${nodePath}\n${historicalRunIds.join("\n")}`,
    [historicalRunIds, nodePath],
  );
  const historicalComparisonQueryKey = useMemo(
    () => `${comparisonNodePath ?? ""}\n${historicalRunIds.join("\n")}`,
    [comparisonNodePath, historicalRunIds],
  );
  const isRunning = activeJob ? RUNNING_STATUSES.has(activeJob.status) : false;
  const monitorCount = activeJob?.monitors.length;
  const isComparing = Boolean(comparisonNodePath);
  const historicalRunGroupEnabled = Boolean(
    enabled && historicalRunGroup && historicalRuns.length > 0,
  );
  const historicalComparisonEnabled =
    historicalRunGroupEnabled && Boolean(comparisonNodePath);
  const singleMonitorEnabled =
    enabled &&
    (activeJob ? activeJob.monitors.length > 0 : Boolean(historicalRun));
  const singleComparisonEnabled = singleMonitorEnabled && isComparing;
  const historicalRequest = useMemo(
    () =>
      createConcurrencyLimiter(
        HISTORICAL_MONITOR_REQUEST_CONCURRENCY,
        historicalRunQueryKey,
      ),
    [historicalRunQueryKey],
  );
  const historicalComparisonRequest = useMemo(
    () =>
      createConcurrencyLimiter(
        HISTORICAL_MONITOR_REQUEST_CONCURRENCY,
        historicalComparisonQueryKey,
      ),
    [historicalComparisonQueryKey],
  );

  const monitorQuery = useQuery<MonitorData>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJob(activeJob.id, nodePath, preset, dataset)
      : monitorQueryKeys.historicalRun(historicalRun?.id, nodePath),
    queryFn: ({ signal }) => {
      if (activeJob) {
        return fetchMonitorData(
          {
            jobId: activeJob.id,
            nodePath,
            preset: preset || undefined,
            dataset: dataset || undefined,
          },
          { signal },
        );
      }
      return fetchLogRunMonitorData(
        {
          runId: historicalRun?.id ?? "",
          nodePath,
        },
        { signal },
      );
    },
    enabled: singleMonitorEnabled,
    retry: false,
    refetchInterval: isRunning ? RUNNING_MONITOR_REFETCH_INTERVAL_MS : false,
  });

  const historicalMonitorQueries = useQueries({
    queries: historicalRuns.map((run) => ({
      queryKey: monitorQueryKeys.historicalRun(run.id, nodePath),
      queryFn: ({ signal }) =>
        historicalRequest(() =>
          fetchLogRunMonitorData(
            {
              runId: run.id,
              nodePath,
            },
            { signal },
          ),
        ),
      enabled: historicalRunGroupEnabled,
      retry: false,
    })),
  });

  const comparisonQuery = useQuery<MonitorData>({
    queryKey: activeJob
      ? monitorQueryKeys.activeJob(
          activeJob.id,
          comparisonNodePath,
          preset,
          dataset,
        )
      : monitorQueryKeys.historicalRun(historicalRun?.id, comparisonNodePath),
    queryFn: ({ signal }) => {
      if (!comparisonNodePath) {
        throw new Error("No comparison node selected");
      }
      if (activeJob) {
        return fetchMonitorData(
          {
            jobId: activeJob.id,
            nodePath: comparisonNodePath,
            preset: preset || undefined,
            dataset: dataset || undefined,
          },
          { signal },
        );
      }
      return fetchLogRunMonitorData(
        {
          runId: historicalRun?.id ?? "",
          nodePath: comparisonNodePath,
        },
        { signal },
      );
    },
    enabled: singleComparisonEnabled,
    retry: false,
    refetchInterval: isRunning ? RUNNING_MONITOR_REFETCH_INTERVAL_MS : false,
  });

  const historicalComparisonQueries = useQueries({
    queries: historicalRuns.map((run) => ({
      queryKey: monitorQueryKeys.historicalRun(run.id, comparisonNodePath),
      queryFn: ({ signal }) =>
        historicalComparisonRequest(() =>
          fetchLogRunMonitorData(
            {
              runId: run.id,
              nodePath: comparisonNodePath ?? "",
            },
            { signal },
          ),
        ),
      enabled: historicalComparisonEnabled,
      retry: false,
    })),
  });

  const historicalData: HistoricalMonitorRunData[] | undefined = historicalRunGroup
    ? historicalRuns.flatMap((run, index) => {
        const data = historicalMonitorQueries[index]?.data;
        return data ? [{ run, data }] : [];
      })
    : undefined;
  const historicalComparisonData: HistoricalMonitorRunData[] | undefined =
    historicalRunGroup && comparisonNodePath
      ? historicalRuns.flatMap((run, index) => {
          const data = historicalComparisonQueries[index]?.data;
          return data ? [{ run, data }] : [];
        })
      : undefined;
  const data: MonitorData | undefined = historicalRunGroup
    ? undefined
    : monitorQuery.data;
  const comparisonData: MonitorData | undefined = historicalRunGroup
    ? undefined
    : comparisonQuery.data;
  const primaryRequests = historicalRunGroupEnabled
    ? historicalMonitorQueries
    : singleMonitorEnabled ? [monitorQuery] : [];
  const comparisonRequests = historicalComparisonEnabled
    ? historicalComparisonQueries
    : singleComparisonEnabled ? [comparisonQuery] : [];
  const primaryFailure = monitorRequestFailure(primaryRequests);
  const comparisonFailure = monitorRequestFailure(comparisonRequests);
  const hasFailure = Boolean(primaryFailure || comparisonFailure);
  const historicalProgress = historicalRunGroup
    ? queryProgress({
        queries: historicalMonitorQueries,
        total: historicalRuns.length,
      })
    : null;
  const historicalComparisonProgress =
    historicalRunGroup && comparisonNodePath
      ? queryProgress({
          queries: historicalComparisonQueries,
          total: historicalRuns.length,
        })
      : null;
  const hasData = historicalRunGroup
    ? isComparing
      ? hasHistoricalMonitorData(historicalData) ||
        hasHistoricalMonitorData(historicalComparisonData)
      : hasHistoricalMonitorData(historicalData)
    : isComparing
      ? hasMonitorData(data) || hasMonitorData(comparisonData)
      : hasMonitorData(data);
  const primaryLoading = primaryRequests.some((query) => query.isFetching);
  const comparisonLoading = comparisonRequests.some((query) => query.isFetching);
  const isFetching = primaryLoading || comparisonLoading;
  const isLoading =
    !hasData &&
    (historicalRunGroup
      ? isFetching
      : primaryRequests.some((query) => query.isLoading) ||
        comparisonRequests.some((query) => query.isLoading));
  const emptyMessage = useMemo(() => {
    if (monitorCount === 0) {
      return {
        title: "No monitor selected",
        detail: "Start a training job with at least one optional monitor enabled.",
      };
    }
    if (isLoading) {
      return {
        title: "Loading monitor data…",
        detail: "Reading TensorBoard event files for the selected node.",
      };
    }
    if (hasFailure) {
      return {
        title: "Could not load monitor data",
        detail: "Retry the failed monitor requests to check for matching tags.",
      };
    }
    if (isRunning) {
      return {
        title: "No data yet",
        detail: "The job is running, but this node has not emitted matching monitor tags yet.",
      };
    }
    if (
      dataWasTruncated(data) ||
      dataWasTruncated(comparisonData) ||
      historicalDataWasTruncated(historicalData) ||
      historicalDataWasTruncated(historicalComparisonData)
    ) {
      return {
        title: "Monitor payload truncated",
        detail:
          truncationReason(data) ??
          truncationReason(comparisonData) ??
          historicalTruncationReason(historicalData) ??
          historicalTruncationReason(historicalComparisonData) ??
          "Some oversized event data was skipped to keep the workbench responsive.",
      };
    }
    if (historicalRunGroup) {
      return {
        title: "No tags for this node",
        detail:
          "No scalar, histogram, or image tags matched this node path in the latest filtered historical runs.",
      };
    }
    if (historicalRun) {
      return {
        title: "No tags for this node",
        detail:
          "No scalar, histogram, or image tags matched this node path in the selected historical run.",
      };
    }
    return {
      title: "No tags for this node",
      detail: "No scalar, histogram, or image tags matched this node path in the selected run.",
    };
  }, [
    comparisonData,
    data,
    historicalComparisonData,
    historicalData,
    historicalRunGroup,
    historicalRun,
    hasFailure,
    isLoading,
    isRunning,
    monitorCount,
  ]);

  const refetch = () => {
    for (const request of [...primaryRequests, ...comparisonRequests]) {
      if (!request.isFetching) {
        void request.refetch();
      }
    }
  };

  return {
    data,
    comparisonData,
    historicalData,
    historicalComparisonData,
    hasData,
    isComparing,
    isFetching,
    isLoading,
    primaryLoading,
    comparisonLoading,
    historicalProgress,
    historicalComparisonProgress,
    emptyMessage,
    primaryFailure,
    comparisonFailure,
    refetch,
  };
}
