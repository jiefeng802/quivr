from typing import Optional
from uuid import UUID

from models.settings import common_dependencies
from models.user_identity import UserIdentity
from pydantic import BaseModel
from repository.user_identity.create_user_identity import (
    create_user_identity,
)


class UserIdentityUpdatableProperties(BaseModel):
    openai_api_key: Optional[str]


def update_user_identity(
    user_id: UUID,
    user_identity_updatable_properties: UserIdentityUpdatableProperties,
) -> UserIdentity:
    commons = common_dependencies()
    response = (
        commons["supabase"]
        .from_("user_identity")
        .update(user_identity_updatable_properties.__dict__)
        .filter("user_id", "eq", user_id)
        .execute()
    )

    if len(response.data) == 0:
        user_identity = UserIdentity(
            user_id=user_id,
            openai_api_key=user_identity_updatable_properties.openai_api_key,
        )
        return create_user_identity(user_identity)

    return UserIdentity(**response.data[0])
