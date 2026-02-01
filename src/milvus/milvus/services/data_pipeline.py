from vila_msgs.msg import CaptionWithPose

from ..models.caption_data import CaptionData
from ..models.vector_data import VectorData
from ..models.milvus_record import MilvusMemoryRecord
from .embedding_service import EmbeddingService


class DataPipeline:
    """Data transformation pipeline for ROS to Milvus."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def process_ros_message(self, msg: CaptionWithPose) -> MilvusMemoryRecord:
        """Transform ROS CaptionWithPose message to MilvusMemoryRecord."""
        caption_data = CaptionData.from_ros_msg(msg)

        vector_data = VectorData.from_caption_data(
            caption_data,
            self.embedding_service.model,
        )

        return MilvusMemoryRecord.from_caption_and_vectors(caption_data, vector_data)
