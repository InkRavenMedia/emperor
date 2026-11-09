from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    ExitStack,
    asynccontextmanager,
)

from fastapi import FastAPI

from emperor_workbench.api._bootstrap import acquire_container
from emperor_workbench.api._container import WorkbenchContainerSlot
from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_lifespan(
    settings: WorkbenchApiSettings,
    *,
    container_slot: WorkbenchContainerSlot,
    project_adapter: ProjectAdapterClient | None,
    training_jobs: TrainingJobService | None = None,
    log_experiment_mutations: LogExperimentMutationCoordinator | None = None,
) -> Lifespan:
    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        container = acquire_container(
            settings,
            project_adapter=project_adapter,
            training_jobs=training_jobs,
            log_experiment_mutations=log_experiment_mutations,
        )
        with ExitStack() as publication:
            async with AsyncExitStack() as resources:
                resources.callback(container.project_adapter.close)
                resources.callback(container.blocking_work.close)
                resources.push_async_callback(container.mutation_execution.close)
                container_slot.install(container)
                publication.callback(container_slot.remove, container)
                api.state.workbench_container = container
                publication.callback(delattr, api.state, "workbench_container")
                yield

    return lifespan


__all__ = ["create_lifespan"]
