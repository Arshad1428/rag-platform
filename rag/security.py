"""Identity: who is asking, which tenant, which groups.

Separation itself is NOT done here: every chunk carries tenant / lateral / ACL tags and every
search filters on them (see rag/providers/index.py). This module only answers "who is this?".

AUTH_MODE:
  demo            pick a user from deploy/demo_users.json (header X-Demo-User). Prototype only.
  entra_easyauth  Azure Container Apps / App Service sign-in (header X-MS-CLIENT-PRINCIPAL).
  oidc            any OpenID Connect provider via a Bearer token: Entra ID, Keycloak, ADFS...
                  Works offline with an on-prem provider (profile C).
  none            no identity at all; everyone is a member of "all" in the default tenant.
"""
import base64
import json
from dataclasses import dataclass, field
from functools import lru_cache

from fastapi import Header, HTTPException

from rag.config import settings


@dataclass(frozen=True)
class User:
    id: str
    tenant: str
    groups: tuple[str, ...] = field(default_factory=tuple)
    name: str = ""

    @property
    def acl_groups(self) -> list[str]:
        """Groups used for filtering: the user's groups plus 'everyone in the tenant'."""
        return sorted(set(self.groups) | {settings.everyone_group})


@lru_cache
def demo_users() -> dict[str, dict]:
    return json.loads(settings.demo_users_file.read_text(encoding="utf-8"))


@lru_cache
def _jwks_client():
    import jwt
    return jwt.PyJWKClient(settings.oidc_jwks_url)


def current_user(
    x_demo_user: str | None = Header(default=None),
    x_ms_client_principal: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> User:
    mode = settings.auth_mode
    if mode == "none":
        return User(id="anonymous", tenant=settings.single_tenant)

    if mode == "demo":
        users = demo_users()
        if not x_demo_user or x_demo_user not in users:
            raise HTTPException(401, "Pick a demo user")
        u = users[x_demo_user]
        return User(id=x_demo_user, tenant=u["tenant"], groups=tuple(u.get("groups", [])),
                    name=u.get("name", x_demo_user))

    if mode == "entra_easyauth":
        if not x_ms_client_principal:
            raise HTTPException(401, "Sign in required")
        p = json.loads(base64.b64decode(x_ms_client_principal))
        claims = p.get("claims", [])
        groups = tuple(c["val"] for c in claims if c.get("typ") == "groups")
        uid = next((c["val"] for c in claims if c.get("typ", "").endswith("objectidentifier")), "user")
        return User(id=uid, tenant=settings.single_tenant, groups=groups)

    if mode == "oidc":
        import jwt
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Sign in required")
        token = authorization.split(" ", 1)[1]
        try:
            key = _jwks_client().get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=settings.oidc_audience,
                                issuer=settings.oidc_issuer)
        except Exception:
            raise HTTPException(401, "Invalid sign-in token")
        groups = claims.get(settings.oidc_groups_claim, [])
        return User(id=claims.get("sub", "user"),
                    tenant=str(claims.get(settings.oidc_tenant_claim, settings.single_tenant)),
                    groups=tuple(groups if isinstance(groups, list) else [groups]),
                    name=claims.get("name", ""))

    raise HTTPException(500, f"Unknown AUTH_MODE {mode}")
