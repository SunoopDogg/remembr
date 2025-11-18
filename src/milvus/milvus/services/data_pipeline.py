"""DataPipeline - orchestrates data transformation pipeline."""

from vila_msgs.msg import CaptionWithPose
from ..models.caption_data import CaptionData
from ..models.vector_data import VectorData
from ..models.milvus_record import MilvusMemoryRecord
from .embedding_service import EmbeddingService


class DataPipeline:
    """Orchestrates data transformation pipeline."""

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize data pipeline.

        Args:
            embedding_service: EmbeddingService instance for generating embeddings
        """
        self.embedding_service = embedding_service

    def process_ros_message(self, msg: CaptionWithPose) -> MilvusMemoryRecord:
        """
        Transform ROS message to Milvus record.

        Pipeline:
        1. ROS msg → CaptionData (extract data)
        2. CaptionData → VectorData (compute embeddings)
        3. CaptionData + VectorData → MilvusMemoryRecord (create record)

        Args:
            msg: ROS CaptionWithPose message

        Returns:
            MilvusMemoryRecord ready for database insertion

        Raises:
            ValueError: If data validation fails
            RuntimeError: If embedding model is not loaded
        """
        # Step 1: Extract data from ROS message
        caption_data = CaptionData.from_ros_msg(msg)

        # Step 2: Compute vector embeddings
        vector_data = VectorData.from_caption_data(
            caption_data,
            self.embedding_service.model
        )

        # Step 3: Create Milvus record
        record = MilvusMemoryRecord.from_caption_and_vectors(
            caption_data,
            vector_data
        )

        return record
