import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TrainingSearchSetup } from "@/features/workbench/components/training-search-setup";
import type { TrainingRunPlanSearch } from "@/features/workbench/state/training/use-training-plan-state";
import type { SearchAxis } from "@/lib/api/models";

const unlockedAxis: SearchAxis = {
  key: "hidden_dim",
  configKey: "HIDDEN_DIM",
  searchKey: "SEARCH_SPACE_HIDDEN_DIM",
  label: "Hidden Dim",
  section: "Model",
  type: "int",
  values: [128, 256],
  locked: false,
};
const lockedAxis: SearchAxis = {
  ...unlockedAxis,
  key: "num_layers",
  configKey: "NUM_LAYERS",
  searchKey: "SEARCH_SPACE_NUM_LAYERS",
  label: "Num Layers",
  values: [2],
  locked: true,
};

function searchFixture(update = vi.fn()): TrainingRunPlanSearch {
  return {
    effective: {
      mode: "grid",
      selectedValues: { hidden_dim: [128] },
      randomSamples: 10,
    },
    axes: [unlockedAxis, lockedAxis],
    isLoading: false,
    conflictKeys: [],
    validation: { ready: true, message: "" },
    lockSummary: {
      lockedAxisCount: 1,
      lockedAxisLabels: ["Num Layers"],
      lockedPresetLabels: ["baseline"],
      lockedAxesMessage: "",
      skippedSelectedAxisCount: 0,
      skippedSelectedAxisLabels: [],
      skippedSelectedAxisMessage: "",
    },
    lockWarning: "",
    modeLabel: "Grid",
    activeAxisCount: 1,
    combinationCount: 1,
    estimatedRunCount: 1,
    unlockedAxisCount: 1,
    unlockedAxes: [unlockedAxis],
    disabledReason: "",
    update,
  };
}

function SearchControls({ search }: { search: TrainingRunPlanSearch }) {
  return (
    <>
      <button type="button">Before search</button>
      <TrainingSearchSetup search={search} />
      <button type="button">After search</button>
    </>
  );
}

describe("TrainingSearchSetup", () => {
  it("skips locked setup controls and re-enables only unlocked axes when training ends", async () => {
    const user = userEvent.setup();
    const search = searchFixture();
    const { rerender } = render(
      <SearchControls
        search={{ ...search, disabledReason: "Training setup is locked while the job is running." }}
      />,
    );

    const axis = screen.getByRole("checkbox", { name: "search axis hidden_dim" });
    const value = screen.getByRole("checkbox", { name: "search value hidden_dim 128" });
    const locked = screen.getByRole("checkbox", { name: "search axis num_layers" });
    expect(axis).toBeDisabled();
    expect(value).toBeDisabled();
    expect(locked).toBeDisabled();
    expect(value).toBeChecked();
    await user.click(axis);
    await user.click(value);
    await user.click(screen.getByRole("button", { name: "Before search" }));
    await user.tab();
    expect(screen.getByRole("button", { name: "After search" })).toHaveFocus();
    expect(search.update).not.toHaveBeenCalled();

    rerender(<SearchControls search={search} />);
    expect(axis).toBeEnabled();
    expect(value).toBeEnabled();
    expect(locked).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "search value num_layers 2" }))
      .toBeDisabled();
    screen.getByRole("button", { name: "Clear" }).focus();
    await user.tab();
    expect(axis).toHaveFocus();
    await user.tab();
    expect(value).toHaveFocus();
    await user.keyboard(" ");
    expect(search.update).toHaveBeenCalledWith({ ...search.effective, selectedValues: {} });
  });

  it("keeps snapshot mode out of the search editing tab sequence", async () => {
    const user = userEvent.setup();
    const search = searchFixture();
    render(
      <SearchControls
        search={{
          ...search,
          effective: { ...search.effective, mode: "off", selectedValues: {} },
          disabledReason: "Config snapshots train fixed variants.",
        }}
      />,
    );

    expect(screen.getByRole("radio", { name: "Grid" })).toBeDisabled();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Before search" }));
    await user.tab();
    expect(screen.getByRole("button", { name: "After search" })).toHaveFocus();
    expect(search.update).not.toHaveBeenCalled();
  });
});
