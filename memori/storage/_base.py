r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""


class BaseStorageAdapter:
    def __init__(self, conn):
        if not callable(conn):
            raise TypeError("conn must be a callable")
        self.conn = conn()

    def close(self):
        if self.conn is not None:
            if hasattr(self.conn, "close"):
                self.conn.close()
            self.conn = None

    def commit(self):
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError

    def get_dialect(self):
        raise NotImplementedError

    def rollback(self):
        raise NotImplementedError


class BaseConversation:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, session_id: int, timeout_minutes: int):
        raise NotImplementedError

    def update(self, id: int, summary: str):
        raise NotImplementedError

    def read(self, id: int) -> dict | None:
        raise NotImplementedError


class BaseConversationMessage:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, conversation_id: int, role: str, type: str, content: str):
        raise NotImplementedError


class BaseConversationMessages:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def read(self, conversation_id: int):
        raise NotImplementedError


class BaseKnowledgeGraph:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, entity_id: int, semantic_triples: list):
        raise NotImplementedError


class BaseEntity:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, external_id: str):
        raise NotImplementedError


class BaseEntityFact:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, entity_id: int, facts: list, fact_embeddings: list | None = None):
        raise NotImplementedError

    def update(self, fact_id: int, new_content: str, new_embedding: list[float]):
        raise NotImplementedError

    def get_embeddings(self, entity_id: int, limit: int = 1000):
        raise NotImplementedError

    def get_facts_by_ids(self, fact_ids: list[int]):
        raise NotImplementedError

    def get_by_id(self, fact_id: int) -> dict | None:
        raise NotImplementedError


class BaseProcess:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, external_id: str):
        raise NotImplementedError


class BaseProcessAttribute:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, process_id: int, attributes: list):
        raise NotImplementedError


class BaseSession:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, uuid: str, entity_id: int, process_id: int):
        raise NotImplementedError


class BaseSchema:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn


class BaseSchemaVersion:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(self, num: int):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError


class BaseExtractionReasoning:
    def __init__(self, conn: BaseStorageAdapter):
        self.conn = conn

    def create(
        self,
        entity_id: int,
        conversation_id: int,
        fact_content: str,
        reasoning: str,
        extraction_type: str,
        decision: str | None = None,
        updated_fact_id: int | None = None,
        embedding_similarity: float | None = None,
    ):
        raise NotImplementedError

    def get_by_conversation(self, conversation_id: int) -> list[dict]:
        raise NotImplementedError