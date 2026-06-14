"""Vendored Vertex AI Vision occupancy protobuf (exact copy of the official
`google/cloud/visionai/v1/annotations.proto`, compiled locally).

Wire-compatible with the real occupancy-analytics model output — no cloud
dependency at runtime. Regenerate with:

    python -m grpc_tools.protoc -Iproto --python_out=proto \\
        proto/visionai_annotations.proto
"""

from .visionai_annotations_pb2 import (  # noqa: F401
    OccupancyCountingPredictionResult,
    StreamAnnotation,
    StreamAnnotations,
    StreamAnnotationType,
    NormalizedPolygon,
    NormalizedPolyline,
    NormalizedVertex,
)

__all__ = [
    "OccupancyCountingPredictionResult",
    "StreamAnnotation",
    "StreamAnnotations",
    "StreamAnnotationType",
    "NormalizedPolygon",
    "NormalizedPolyline",
    "NormalizedVertex",
]
