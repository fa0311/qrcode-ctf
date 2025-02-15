from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def all_schema():
        value = {
            subclass.__name__: (subclass, ...) for subclass in Base.__subclasses__()
        }
        model: BaseModel = create_model(
            Base.__name__,
            __config__=Base.model_config,
            **value,  # type: ignore
        )
        return model.model_json_schema(mode="serialization")


class FlagDescription(Base):
    ja: str = Field()
    en: str = Field()


class FlagModelOne(Base):
    name: str = Field()
    description: FlagDescription = Field()
    key: str = Field()


class FlagModelMulti(Base):
    name: str = Field()
    description: FlagDescription = Field()
    key: list[str] = Field()


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ctf_seed: str = Field()

    def __init__(self):
        super().__init__()
