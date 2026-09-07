"""SQLAlchemy models for datasets and algorithm experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[str] = mapped_column(String(64), default="1")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    generator_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    images: Mapped[list["ImageRecord"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="dataset")

    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_name_version"),)


class ImageRecord(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str] = mapped_column(Text)
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    split: Mapped[str] = mapped_column(String(32), default="test")
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    dataset: Mapped[Dataset] = relationship(back_populates="images")
    ground_truth_objects: Mapped[list["GroundTruthObject"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="image")

    __table_args__ = (UniqueConstraint("dataset_id", "file_path", name="uq_dataset_image_path"),)


class GroundTruthObject(Base):
    __tablename__ = "ground_truth_objects"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    object_index: Mapped[int] = mapped_column(Integer)
    mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)
    axis_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    directed_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    class_name: Mapped[str] = mapped_column(String(200), default="object")
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    image: Mapped[ImageRecord] = relationship(back_populates="ground_truth_objects")
    __table_args__ = (UniqueConstraint("image_id", "object_index", name="uq_gt_image_object"),)


class Algorithm(Base):
    __tablename__ = "algorithms"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="algorithm")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    algorithm_id: Mapped[int] = mapped_column(ForeignKey("algorithms.id"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    git_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)

    algorithm: Mapped[Algorithm] = relationship(back_populates="runs")
    dataset: Mapped[Dataset] = relationship(back_populates="runs")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    object_index: Mapped[int] = mapped_column(Integer)
    mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)
    axis_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    directed_angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    bbox: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    run: Mapped[ExperimentRun] = relationship(back_populates="predictions")
    image: Mapped[ImageRecord] = relationship(back_populates="predictions")
    __table_args__ = (UniqueConstraint("run_id", "image_id", "object_index", name="uq_prediction_object"),)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id", ondelete="CASCADE"), index=True)
    image_id: Mapped[int | None] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), nullable=True, index=True)
    object_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)

    run: Mapped[ExperimentRun] = relationship(back_populates="metrics")
