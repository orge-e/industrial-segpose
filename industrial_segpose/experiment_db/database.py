"""Database initialization and repository-style write/query helpers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, selectinload

from .models import (
    Algorithm,
    Base,
    Dataset,
    ExperimentRun,
    GroundTruthObject,
    ImageRecord,
    Metric,
    Prediction,
)


def default_database_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / "data" / "benchmark" / "segpose_benchmark.db"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentDatabase:
    """Small transactional API shared by generators and benchmark runners."""

    def __init__(self, path_or_url: str | Path | None = None, *, echo: bool = False):
        value = path_or_url or default_database_path()
        if isinstance(value, Path) or "://" not in str(value):
            path = Path(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.path: Path | None = path
            url = f"sqlite:///{path.resolve().as_posix()}"
        else:
            self.path = None
            url = str(value)
        self.engine: Engine = create_engine(url, echo=echo, future=True)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def rebuild(self) -> None:
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine, expire_on_commit=False) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def create_dataset(
        self,
        name: str,
        source_type: str,
        *,
        version: str = "1",
        description: str | None = None,
        generator_config: dict[str, Any] | None = None,
    ) -> Dataset:
        if source_type not in {"synthetic", "public", "historical", "real"}:
            raise ValueError(f"Unsupported dataset source_type: {source_type}")
        with self.session() as session:
            item = Dataset(
                name=name,
                source_type=source_type,
                version=version,
                description=description,
                generator_config=generator_config,
            )
            session.add(item)
            session.flush()
            session.expunge(item)
            return item

    def register_image(
        self,
        dataset_id: int,
        file_path: str | Path,
        width: int,
        height: int,
        *,
        split: str = "test",
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        file_hash: str | None = None,
    ) -> ImageRecord:
        path = Path(file_path)
        digest = file_hash or file_sha256(path)
        with self.session() as session:
            item = ImageRecord(
                dataset_id=dataset_id,
                file_path=str(path),
                file_hash=digest,
                width=width,
                height=height,
                split=split,
                source=source,
                metadata_json=metadata,
            )
            session.add(item)
            session.flush()
            session.expunge(item)
            return item

    def add_ground_truth(
        self,
        image_id: int,
        object_index: int,
        *,
        center_x: float,
        center_y: float,
        mask_path: str | Path | None = None,
        axis_angle_deg: float | None = None,
        directed_angle_deg: float | None = None,
        bbox: list[float] | tuple[float, ...] | None = None,
        class_name: str = "object",
        metadata: dict[str, Any] | None = None,
    ) -> GroundTruthObject:
        with self.session() as session:
            item = GroundTruthObject(
                image_id=image_id,
                object_index=object_index,
                mask_path=str(mask_path) if mask_path is not None else None,
                center_x=center_x,
                center_y=center_y,
                axis_angle_deg=axis_angle_deg,
                directed_angle_deg=directed_angle_deg,
                bbox=list(bbox) if bbox is not None else None,
                class_name=class_name,
                metadata_json=metadata,
            )
            session.add(item)
            session.flush()
            session.expunge(item)
            return item

    def get_or_create_algorithm(self, name: str, description: str | None = None) -> Algorithm:
        with self.session() as session:
            existing = session.scalar(select(Algorithm).where(Algorithm.name == name))
            if existing is None:
                existing = Algorithm(name=name, description=description)
                session.add(existing)
                session.flush()
            session.expunge(existing)
            return existing

    def create_run(
        self,
        algorithm_id: int,
        dataset_id: int,
        config: dict[str, Any],
        *,
        git_state: str | None = None,
    ) -> ExperimentRun:
        with self.session() as session:
            item = ExperimentRun(
                algorithm_id=algorithm_id,
                dataset_id=dataset_id,
                config=config,
                git_state=git_state,
                status="running",
            )
            session.add(item)
            session.flush()
            session.expunge(item)
            return item

    def finish_run(self, run_id: int, status: str = "completed") -> None:
        with self.session() as session:
            run = session.get(ExperimentRun, run_id)
            if run is None:
                raise KeyError(f"Experiment run does not exist: {run_id}")
            run.status = status
            run.finished_at = datetime.now(timezone.utc)

    def add_prediction(self, run_id: int, image_id: int, object_index: int, **values) -> Prediction:
        with self.session() as session:
            item = Prediction(run_id=run_id, image_id=image_id, object_index=object_index, **values)
            session.add(item)
            session.flush()
            session.expunge(item)
            return item

    def add_metric(
        self,
        run_id: int,
        name: str,
        value: float | None,
        *,
        image_id: int | None = None,
        object_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Metric:
        with self.session() as session:
            item = Metric(
                run_id=run_id,
                image_id=image_id,
                object_index=object_index,
                name=name,
                value=value,
                metadata_json=metadata,
            )
            session.add(item)
            session.flush()
            session.expunge(item)
            return item

    def datasets(self) -> list[Dataset]:
        with self.session() as session:
            return list(session.scalars(select(Dataset).order_by(Dataset.created_at)).all())

    def images_for_dataset(self, dataset_id: int) -> list[ImageRecord]:
        with self.session() as session:
            statement = (
                select(ImageRecord)
                .where(ImageRecord.dataset_id == dataset_id)
                .options(selectinload(ImageRecord.ground_truth_objects))
                .order_by(ImageRecord.id)
            )
            return list(session.scalars(statement).all())

    def runs_for_dataset(self, dataset_id: int) -> list[ExperimentRun]:
        with self.session() as session:
            statement = (
                select(ExperimentRun)
                .where(ExperimentRun.dataset_id == dataset_id)
                .options(
                    selectinload(ExperimentRun.algorithm),
                    selectinload(ExperimentRun.predictions),
                    selectinload(ExperimentRun.metrics),
                )
                .order_by(ExperimentRun.started_at)
            )
            return list(session.scalars(statement).all())
