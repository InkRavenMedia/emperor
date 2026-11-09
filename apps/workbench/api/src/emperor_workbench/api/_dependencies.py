from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from emperor_workbench.api._container import (
    WorkbenchContainer,
    WorkbenchContainerSlot,
)
from emperor_workbench.config_snapshots import ConfigSnapshotService
from emperor_workbench.inspection import InspectionService
from emperor_workbench.project_adapter import ProjectAdapterClient
from emperor_workbench.run_history import RunHistoryService
from emperor_workbench.run_plans import RunPlanService
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import TrainingJobService


async def get_workbench_container(request: Request) -> WorkbenchContainer:
    slot = cast(
        WorkbenchContainerSlot,
        request.app.state.workbench_container_slot,
    )
    return slot.get()


_WorkbenchContainerDep = Annotated[WorkbenchContainer, Depends(get_workbench_container)]


async def get_workbench_settings(
    services: _WorkbenchContainerDep,
) -> WorkbenchApiSettings:
    return services.settings


async def get_project_adapter_client(
    services: _WorkbenchContainerDep,
) -> ProjectAdapterClient:
    return services.project_adapter


async def get_config_snapshot_service(
    services: _WorkbenchContainerDep,
) -> ConfigSnapshotService:
    return services.config_snapshots


async def get_inspection_service(
    services: _WorkbenchContainerDep,
) -> InspectionService:
    return services.inspection


async def get_run_history_service(
    services: _WorkbenchContainerDep,
) -> RunHistoryService:
    return services.run_history


async def get_training_job_service(
    services: _WorkbenchContainerDep,
) -> TrainingJobService:
    return services.training_jobs


async def get_training_run_plan_service(
    services: _WorkbenchContainerDep,
) -> RunPlanService:
    return services.training_run_plans


ProjectAdapterDep = Annotated[ProjectAdapterClient, Depends(get_project_adapter_client)]
TrainingJobServiceDep = Annotated[TrainingJobService, Depends(get_training_job_service)]
WorkbenchSettingsDep = Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)]


__all__ = [
    "ProjectAdapterDep",
    "TrainingJobServiceDep",
    "WorkbenchSettingsDep",
    "get_workbench_container",
    "get_config_snapshot_service",
    "get_inspection_service",
    "get_project_adapter_client",
    "get_run_history_service",
    "get_training_job_service",
    "get_training_run_plan_service",
    "get_workbench_settings",
]
