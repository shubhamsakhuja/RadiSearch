# =============================================================================
# auth.py — Microsoft Azure AD Single Sign-On Authentication
# =============================================================================
#
# WHAT THIS MODULE DOES:
#   Implements the complete Microsoft OAuth2 / OpenID Connect login flow
#   using the MSAL (Microsoft Authentication Library) package.
#   Registers all auth-related routes as a Flask Blueprint.
#   Provides the @login_required decorator used by every other route.
#
# THE LOGIN FLOW (plain English):
#   1. User visits any protected page (e.g. /)
#   2. @login_required checks Flask session — no "user" key found
#   3. User is redirected to /auth/login
#   4. /auth/login generates a random state token and redirects the browser
#      to Microsoft's login page (login.microsoftonline.com)
#   5. User types their @hospital.com credentials on Microsoft's servers
#      — we never see the password
#   6. Microsoft redirects back to /auth/callback with a short-lived code
#   7. /auth/callback exchanges the code for a token (server-to-server)
#   8. We verify the token and check the email ends in @hospital.com
#   9. User info (name + email) is stored in the encrypted Flask session cookie
#  10. User is redirected to the page they originally wanted
#
# SECURITY PROPERTIES:
#   - Passwords never touch our server
#   - Session cookie is HttpOnly (JS cannot read it) + SameSite=Lax
#   - Random state token prevents CSRF attacks during the OAuth flow
#   - Domain check ensures only @hospital.com accounts are admitted
#   - /auth/logout clears both our session AND the Microsoft SSO session
#
# PUBLIC API:
#   auth_bp          — Flask Blueprint to register in app.py
#   login_required   — decorator to protect any route
#
# SETUP (IT team, ~10 minutes in Azure Portal):
#   1. portal.azure.com → Azure Active Directory → App registrations → New
#   2. Name: "RadiSearch"
#   3. Redirect URI: http://<server-ip>:5000/auth/callback
#   4. Copy Application (client) ID  → AZURE_CLIENT_ID  in .env
#   5. Copy Directory (tenant) ID    → AZURE_TENANT_ID  in .env
#   6. Certificates & secrets → New client secret → copy Value
#                              → AZURE_CLIENT_SECRET in .env
#   7. Add AZURE_ALLOWED_DOMAIN=hospital.com and FLASK_SECRET_KEY to .env
# =============================================================================

import uuid
import logging
from functools import wraps

import msal
from flask import (
    Blueprint,
    request,
    session,
    redirect,
    url_for,
    render_template,
)

from config import (
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
    AZURE_AUTHORITY,
    AZURE_SCOPE,
    AZURE_ALLOWED_DOMAIN,
    DEV_MODE,
)


# =============================================================================
# BLUEPRINT
# =============================================================================
# A Flask Blueprint is a self-contained collection of routes.
# We register it in app.py with: app.register_blueprint(auth_bp)
# All routes defined here get the prefix defined at registration time.
# Auth routes have no prefix — /auth/login, /auth/callback etc. are their
# full paths as written.

auth_bp = Blueprint("auth", __name__)


# =============================================================================
# MSAL APP FACTORY
# =============================================================================

def _get_msal_app() -> msal.ConfidentialClientApplication:
    """
    Creates and returns a fresh MSAL ConfidentialClientApplication.

    We create a new instance per request rather than caching one globally
    because MSAL manages its own token cache internally. A fresh instance
    per request is the safest pattern for a multi-user web app — it avoids
    any risk of one user's token cache leaking to another user's session.

    "Confidential" client means our app has a client secret (as opposed to
    a "Public" client like a mobile app which can't safely store secrets).
    This is the correct type for a server-side web application.
    """
    return msal.ConfidentialClientApplication(
        AZURE_CLIENT_ID,
        authority=AZURE_AUTHORITY,
        client_credential=AZURE_CLIENT_SECRET,
    )


def _build_redirect_uri() -> str:
    """
    Builds the OAuth2 callback URL dynamically from the current request host.
    e.g. http://192.168.1.50:5000/auth/callback

    Using url_for() with _external=True means this works correctly regardless
    of which IP address or hostname the server is accessed from — useful when
    the server IP changes or when testing on localhost vs the intranet IP.
    """
    return url_for("auth.callback", _external=True)


# =============================================================================
# LOGIN REQUIRED DECORATOR
# =============================================================================

def login_required(f):
    """
    Decorator that protects any Flask route from unauthenticated access.

    DEV MODE (DEV_MODE=true in .env):
        Skips all authentication and auto-injects a fake "Dev User" session.
        Use this for local testing without Azure AD credentials.
        NEVER enable in production — it disables all access control.

    PRODUCTION MODE (default):
        Checks for a "user" key in the Flask session. If not found, saves
        the requested URL and redirects to /auth/login.
        After a successful Microsoft login, the user is redirected back
        to the page they originally wanted.

    Usage:
        @app.route("/")
        @login_required
        def home():
            return render_template("landing.html")
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # ── DEV MODE: skip auth, inject fake user ─────────────────────────────
        if DEV_MODE:
            if "user" not in session:
                session["user"] = {
                    "name":  "Dev User",
                    "email": "dev@localhost",
                }
            return f(*args, **kwargs)

        # ── PRODUCTION: require valid Microsoft login ─────────────────────────
        if "user" not in session:
            session["next"] = request.url
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# AUTH ROUTES
# =============================================================================

@auth_bp.route("/auth/login")
def login():
    """
    Step 1 of the OAuth2 flow.

    In DEV MODE: immediately injects a fake session and redirects home —
    no Microsoft interaction at all.

    In production: generates a CSRF state token, builds the Microsoft
    login URL, and redirects the browser there.
    """
    # ── DEV MODE: skip Microsoft entirely ────────────────────────────────────
    if DEV_MODE:
        session["user"] = {"name": "Dev User", "email": "dev@localhost"}
        logging.info("[auth] DEV MODE — auto-logged in as Dev User.")
        next_url = session.pop("next", url_for("pages.home"))
        return redirect(next_url)

    # ── PRODUCTION: redirect to Microsoft login ───────────────────────────────
    state = str(uuid.uuid4())
    session["oauth_state"] = state

    auth_url = _get_msal_app().get_authorization_request_url(
        scopes=AZURE_SCOPE,
        state=state,
        redirect_uri=_build_redirect_uri(),
    )

    logging.info(f"[auth] Redirecting to Microsoft login. State: {state[:8]}...")
    return redirect(auth_url)


@auth_bp.route("/auth/callback")
def callback():
    """
    Step 2 of the OAuth2 flow — Microsoft redirects here after login.

    Microsoft sends back two query parameters:
      code  — a short-lived authorisation code (expires in ~10 minutes)
      state — the random token we sent in Step 1

    This function:
      1. Checks for errors from Microsoft (user cancelled, etc.)
      2. Verifies the state token matches (CSRF protection)
      3. Exchanges the code for an access token (server-to-server HTTPS call)
      4. Extracts user email and name from the token claims
      5. Verifies the email domain is allowed (@hospital.com)
      6. Stores user info in the Flask session
      7. Redirects to the page the user originally wanted
    """
    # ── Error from Microsoft ──────────────────────────────────────────────────
    # e.g. user clicked "Cancel" on the login page
    if "error" in request.args:
        detail = request.args.get("error_description", "Unknown error from Microsoft.")
        logging.warning(f"[auth] Microsoft returned error: {detail}")
        return render_template(
            "auth_error.html",
            error="Microsoft login failed.",
            detail=detail,
        ), 401

    # ── CSRF state token verification ─────────────────────────────────────────
    # If the state doesn't match what we stored, this could be a CSRF attack —
    # someone tricking the browser into completing a login flow they started.
    received_state = request.args.get("state", "")
    expected_state = session.get("oauth_state", "")

    if received_state != expected_state:
        logging.warning(
            f"[auth] State mismatch — possible CSRF attempt. "
            f"Received: {received_state[:8]}... Expected: {expected_state[:8]}..."
        )
        return render_template(
            "auth_error.html",
            error="Security check failed.",
            detail="Login state mismatch. Please try logging in again.",
        ), 403

    # ── Exchange code for token ───────────────────────────────────────────────
    # This is a server-to-server HTTPS call from our Flask app to Microsoft.
    # The browser is not involved — the "code" is only usable once and expires.
    code = request.args.get("code", "")
    result = _get_msal_app().acquire_token_by_authorization_code(
        code,
        scopes=AZURE_SCOPE,
        redirect_uri=_build_redirect_uri(),
    )

    if "error" in result:
        detail = result.get("error_description", "Token exchange failed.")
        logging.error(f"[auth] Token exchange error: {detail}")
        return render_template(
            "auth_error.html",
            error="Could not complete login.",
            detail=detail,
        ), 401

    # ── Extract user profile ──────────────────────────────────────────────────
    # id_token_claims contains the user's profile decoded from the JWT token.
    # "preferred_username" is typically the user's email in Azure AD.
    # We fall back to "email" for tenants that configure it differently.
    claims = result.get("id_token_claims", {})
    email  = (claims.get("preferred_username") or claims.get("email") or "").lower()
    name   = claims.get("name", email)  # Full display name, e.g. "Dr Jane Smith"

    if not email:
        logging.error("[auth] Could not extract email from token claims.")
        return render_template(
            "auth_error.html",
            error="Login incomplete.",
            detail="Could not retrieve your email address from Microsoft. "
                   "Contact your IT administrator.",
        ), 401

    # ── Domain check ─────────────────────────────────────────────────────────
    # Only allow emails from the configured organisation domain.
    # This is the critical gate — even a valid Microsoft account from another
    # organisation (e.g. a personal hotmail.com) will be blocked here.
    if AZURE_ALLOWED_DOMAIN and not email.endswith(f"@{AZURE_ALLOWED_DOMAIN}"):
        logging.warning(f"[auth] Access denied — wrong domain: {email}")
        return render_template(
            "auth_error.html",
            error="Access Denied",
            detail=(
                f"Only @{AZURE_ALLOWED_DOMAIN} accounts are permitted. "
                f"You attempted to log in as {email}. "
                "Contact your IT administrator if you believe this is an error."
            ),
        ), 403

    # ── Store in session ──────────────────────────────────────────────────────
    # Flask's session is an encrypted, signed cookie stored in the browser.
    # Contents can only be read/verified by the server (which holds the secret key).
    session["user"] = {"email": email, "name": name}
    session.pop("oauth_state", None)  # Clean up — no longer needed

    logging.info(f"[auth] Login successful: {email}")

    # Redirect to the page they originally wanted, or home if none was saved
    next_url = session.pop("next", url_for("pages.home"))
    return redirect(next_url)


@auth_bp.route("/auth/logout")
def logout():
    """
    Logs the user out of both the RadiSearch app and their Microsoft session.

    Two-stage logout:
      1. Clear the Flask session — removes our login cookie. The user is
         immediately unauthenticated in this app.
      2. Redirect to Microsoft's logout endpoint — clears the Microsoft
         SSO session. Without this step, visiting /auth/login again would
         auto-log them back in without asking for credentials again.

    post_logout_redirect_uri tells Microsoft where to send the user after
    their Microsoft session is cleared — we send them to /auth/loggedout
    which shows a clean "you've been logged out" confirmation page.
    """
    user_email = session.get("user", {}).get("email", "unknown")
    logging.info(f"[auth] Logging out: {user_email}")

    session.clear()

    # Build Microsoft's logout URL
    # After clearing the Microsoft session, the user is redirected to our
    # /auth/loggedout page (the post_logout_redirect_uri must be registered
    # in the Azure Portal under "Redirect URIs" for logout to work correctly)
    logged_out_url = url_for("auth.loggedout", _external=True)
    microsoft_logout_url = (
        f"{AZURE_AUTHORITY}/oauth2/v2.0/logout"
        f"?post_logout_redirect_uri={logged_out_url}"
    )
    return redirect(microsoft_logout_url)


@auth_bp.route("/auth/loggedout")
def loggedout():
    """
    Simple confirmation page shown after a successful logout.
    No login_required — must be publicly accessible so Microsoft can
    redirect here after clearing the SSO session.
    """
    return render_template("logged_out.html")