#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confluence Wiki Subpage Parser
- Parses child pages (not linked pages) using XPath
- Converts HTML content to Markdown
- Persists all links (internal parsed, internal unparsed, external)
- Handles authentication via cookies
- Configurable depth limit

Configuration via:
  1. Environment variables
  2. .env file
  3. config.json
  4. Interactive prompts
"""

import os
import re
import json
import time
import logging
import hashlib
import getpass
from urllib.parse import urljoin, urlparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import requests
from lxml import etree
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG_FILE = "config.json"
ENV_FILE = ".env"

DEFAULT_CONFIG = {
    "confluence_base_url": "",  # e.g. https://mycompany.atlassian.net/wiki
    "start_page_url": "",  # e.g. https://mycompany.atlassian.net/wiki/spaces/TEAM/pages/12345/Home
    "username": "",  # email for Cloud, username for Server
    "password": "",  # API token for Cloud, password for Server
    "max_depth": 3,
    "output_dir": "confluence_export",
    "delay_between_requests": 1,
    "log_level": "INFO",
}

LINKS_FILE = "link_registry.json"


# ─── Config Loader ───────────────────────────────────────────────────────────


def load_env_file(filepath: str) -> dict:
    """Parse a .env file into a dict."""
    env_vars = {}
    if not os.path.exists(filepath):
        return env_vars
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                env_vars[key] = value
    return env_vars


def load_config() -> dict:
    """
    Load configuration from multiple sources (later overrides earlier):
      1. Defaults
      2. config.json
      3. .env file
      4. OS environment variables
      5. Interactive prompt (for missing required fields)
    """
    config = dict(DEFAULT_CONFIG)

    # --- From config.json ---
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                file_config = json.load(f)
            config.update({k: v for k, v in file_config.items() if v})
            print(f"[config] Loaded from {CONFIG_FILE}")
        except Exception as e:
            print(f"[config] Warning: could not read {CONFIG_FILE}: {e}")

    # --- From .env file ---
    env_vars = load_env_file(ENV_FILE)
    env_mapping = {
        "CONFLUENCE_BASE_URL": "confluence_base_url",
        "CONFLUENCE_START_URL": "start_page_url",
        "CONFLUENCE_USERNAME": "username",
        "CONFLUENCE_PASSWORD": "password",
        "CONFLUENCE_MAX_DEPTH": "max_depth",
        "CONFLUENCE_OUTPUT_DIR": "output_dir",
    }
    for env_key, config_key in env_mapping.items():
        val = env_vars.get(env_key) or os.environ.get(env_key)
        if val:
            if config_key == "max_depth":
                config[config_key] = int(val)
            else:
                config[config_key] = val

    # --- Interactive prompts for missing required fields ---
    if not config["confluence_base_url"]:
        config["confluence_base_url"] = input(
            "Enter Confluence base URL\n"
            "  (e.g. https://mycompany.atlassian.net/wiki): "
        ).strip().rstrip("/")

    if not config["start_page_url"]:
        config["start_page_url"] = input(
            "Enter the starting page URL\n"
            "  (e.g. https://mycompany.atlassian.net/wiki/spaces/TEAM/pages/12345/Home): "
        ).strip()

    if not config["username"]:
        config["username"] = input(
            "Enter your Confluence username (email for Cloud): "
        ).strip()

    if not config["password"]:
        config["password"] = getpass.getpass(
            "Enter your Confluence password/API token: "
        )

    # --- Validate ---
    base = config["confluence_base_url"]
    start = config["start_page_url"]
    if not base.startswith("http"):
        print(f"[config] ERROR: Invalid base URL: {base}")
        raise SystemExit(1)
    if not start.startswith("http"):
        print(f"[config] ERROR: Invalid start URL: {start}")
        raise SystemExit(1)

    # --- Offer to save for next run ---
    if not os.path.exists(CONFIG_FILE):
        save = input("Save config to config.json for next run? (y/n): ").strip().lower()
        if save == "y":
            save_config = dict(config)
            save_config.pop("password", None)  # Never save password to file
            with open(CONFIG_FILE, "w") as f:
                json.dump(save_config, f, indent=2)
            print(f"[config] Saved to {CONFIG_FILE} (password NOT saved)")
            print(f"[config] Tip: set CONFLUENCE_PASSWORD env var or use .env file")

    return config


# ─── Logging Setup ───────────────────────────────────────────────────────────


def setup_logging(level_str: str = "INFO"):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("confluence_parser.log"),
        ],
    )
    return logging.getLogger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class LinkEntry:
    url: str
    title: str = ""
    parsed: bool = False
    is_child: bool = False
    is_external: bool = False
    local_file: str = ""
    found_on_pages: list = field(default_factory=list)


@dataclass
class PageInfo:
    url: str
    title: str
    page_id: str
    space_key: str
    depth: int
    parent_url: str = ""
    children_urls: list = field(default_factory=list)
    local_file: str = ""


# ─── Link Registry ──────────────────────────────────────────────────────────


class LinkRegistry:
    """Persists all discovered links so parsing can be resumed."""

    def __init__(self, filepath: str, logger_inst):
        self.filepath = filepath
        self.logger = logger_inst
        self.links: dict[str, LinkEntry] = {}
        self.parsed_pages: dict[str, PageInfo] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for url, entry in data.get("links", {}).items():
                    self.links[url] = LinkEntry(**entry)
                for url, page in data.get("parsed_pages", {}).items():
                    self.parsed_pages[url] = PageInfo(**page)
                self.logger.info(
                    f"Loaded registry: {len(self.links)} links, "
                    f"{len(self.parsed_pages)} parsed pages"
                )
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Could not load registry: {e}. Starting fresh.")

    def save(self):
        data = {
            "links": {url: asdict(entry) for url, entry in self.links.items()},
            "parsed_pages": {url: asdict(p) for url, p in self.parsed_pages.items()},
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.debug(f"Registry saved: {len(self.links)} links")

    def register_link(
            self,
            url: str,
            title: str = "",
            parsed: bool = False,
            is_child: bool = False,
            is_external: bool = False,
            local_file: str = "",
            found_on: str = "",
    ):
        url = self._normalize_url(url)
        if url in self.links:
            entry = self.links[url]
            if title:
                entry.title = title
            if parsed:
                entry.parsed = True
            if is_child:
                entry.is_child = True
            if local_file:
                entry.local_file = local_file
            if found_on and found_on not in entry.found_on_pages:
                entry.found_on_pages.append(found_on)
        else:
            self.links[url] = LinkEntry(
                url=url,
                title=title,
                parsed=parsed,
                is_child=is_child,
                is_external=is_external,
                local_file=local_file,
                found_on_pages=[found_on] if found_on else [],
            )

    def register_parsed_page(self, page_info: PageInfo):
        self.parsed_pages[self._normalize_url(page_info.url)] = page_info

    def is_parsed(self, url: str) -> bool:
        url = self._normalize_url(url)
        return url in self.parsed_pages

    def get_unparsed_internal_links(self) -> list[LinkEntry]:
        return [
            entry
            for entry in self.links.values()
            if not entry.parsed and not entry.is_external
        ]

    def get_local_file_for_url(self, url: str) -> Optional[str]:
        url = self._normalize_url(url)
        if url in self.links and self.links[url].local_file:
            return self.links[url].local_file
        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url)
        normalized = parsed._replace(fragment="")
        result = normalized.geturl().rstrip("/")
        return result


# ─── Confluence Session ──────────────────────────────────────────────────────


class ConfluenceSession:
    """Handles authentication and HTTP requests with cookie persistence."""

    COOKIE_FILE = ".confluence_cookies.json"

    def __init__(self, base_url: str, username: str, password: str, logger_inst):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.logger = logger_inst
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; ConfluenceParser/1.0)"}
        )
        self._is_cloud = "atlassian.net" in self.base_url
        self._load_cookies()

    def _load_cookies(self):
        if os.path.exists(self.COOKIE_FILE):
            try:
                with open(self.COOKIE_FILE, "r") as f:
                    cookies = json.load(f)
                for name, value in cookies.items():
                    self.session.cookies.set(name, value)
                self.logger.info(f"Loaded saved cookies ({len(cookies)} cookies)")
            except Exception as e:
                self.logger.warning(f"Could not load cookies: {e}")

    def _save_cookies(self):
        cookies = {c.name: c.value for c in self.session.cookies}
        with open(self.COOKIE_FILE, "w") as f:
            json.dump(cookies, f)
        self.logger.debug(f"Cookies saved ({len(cookies)} cookies)")

    def login(self) -> bool:
        """
        Authenticate with Confluence. Tries in order:
          1. Existing cookies
          2. Basic auth (Cloud + API token)
          3. Form-based login (Server/Data Center)
        """
        # --- Try existing session ---
        if self._test_auth():
            self.logger.info("Existing session is valid")
            return True

        # --- Cloud: Basic auth with API token ---
        if self._is_cloud:
            self.logger.info(
                f"Attempting Atlassian Cloud API auth as '{self.username}'..."
            )
            self.session.auth = (self.username, self.password)
            if self._test_auth():
                self._save_cookies()
                self.logger.info("Cloud API token authentication successful")
                return True
            self.logger.warning("Cloud API auth failed. "
                                "Make sure you use an API token, not your password.")
            self.logger.warning(
                "Generate one at: https://id.atlassian.com/manage-profile/security/api-tokens"
            )
            self.session.auth = None

        # --- Server: Form-based login ---
        self.logger.info(f"Attempting form-based login as '{self.username}'...")
        login_urls = [
            f"{self.base_url}/dologin.action",
            f"{self.base_url}/login.action",
        ]
        for login_url in login_urls:
            try:
                # First GET the login page to pick up any CSRF tokens
                login_page = self.session.get(login_url, timeout=15)

                # Check if this is an SSO/OIDC redirect page
                if self._is_sso_redirect_page(login_page.text):
                    self.logger.warning(
                        "SSO/OIDC authentication detected. "
                        "Please log in manually in a browser, then export cookies."
                    )
                    self._save_cookies()
                    # Try auth again after saving cookies (user may have logged in manually)
                    if self._test_auth():
                        return True
                    # Continue with form login attempt anyway
                    continue

                # Extract hidden form fields (CSRF token, etc.)
                hidden_fields = {}
                try:
                    tree = etree.HTML(login_page.text)
                    for inp in tree.xpath('//form//input[@type="hidden"]'):
                        name = inp.get("name")
                        value = inp.get("value", "")
                        if name:
                            hidden_fields[name] = value
                except Exception:
                    pass

                login_data = {
                    **hidden_fields,
                    "os_username": self.username,
                    "os_password": self.password,
                    "os_destination": "",
                    "login": "Log In",
                }

                resp = self.session.post(
                    login_url,
                    data=login_data,
                    allow_redirects=True,
                    timeout=15,
                )

                # Check if we ended up on a non-login page
                if resp.status_code == 200:
                    resp_lower = resp.url.lower()
                    if (
                            "login" not in resp_lower
                            and "authenticate" not in resp_lower
                    ):
                        self._save_cookies()
                        self.logger.info(
                            f"Form-based login successful via {login_url}"
                        )
                        return True
                    # Also check if REST API now works
                    if self._test_auth():
                        self._save_cookies()
                        self.logger.info("Form login successful (verified via REST)")
                        return True

            except requests.exceptions.ConnectionError:
                self.logger.debug(f"Could not connect to {login_url}")
            except Exception as e:
                self.logger.debug(f"Login attempt to {login_url} failed: {e}")

        self.logger.error(
            "All authentication methods failed.\n"
            "For Atlassian Cloud: use your email + API token\n"
            "  (https://id.atlassian.com/manage-profile/security/api-tokens)\n"
            "For Confluence Server: use your username + password\n"
            "For SSO/OIDC: Log in manually in a browser first, then run the script."
        )
        return False

    def _is_sso_redirect_page(self, html_content: str) -> bool:
        """Check if the page is an SSO/OIDC redirect page."""
        sso_indicators = [
            "window.location.assign",
            "window.location.href",
            "oidc",
            "openid",
            "/auth/",
            "saml",
            "sso",
            "authenticate",
        ]
        html_lower = html_content.lower()
        return any(indicator in html_lower for indicator in sso_indicators)

    def _test_auth(self) -> bool:
        """Test if current session is authenticated by calling a REST endpoint."""
        test_urls = [
            f"{self.base_url}/rest/api/user/current",
            f"{self.base_url}/rest/api/space?limit=1",
        ]
        for test_url in test_urls:
            try:
                resp = self.session.get(test_url, timeout=10)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        display = data.get("displayName") or data.get("username", "")
                        if display:
                            self.logger.info(f"Authenticated as: {display}")
                    except Exception:
                        pass
                    return True
            except Exception:
                continue
        return False

    def get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make an authenticated GET request with auto-retry on 401."""
        try:
            resp = self.session.get(url, timeout=30, **kwargs)
            if resp.status_code == 401:
                self.logger.warning("Got 401, re-authenticating...")
                if self.login():
                    resp = self.session.get(url, timeout=30, **kwargs)
                else:
                    return None
            resp.raise_for_status()
            self._save_cookies()
            return resp
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error fetching {url}: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error fetching {url}: {e}")
        return None

    def get_page_by_rest_api(self, page_id: str) -> Optional[dict]:
        """Fetch page content via REST API."""
        url = (
            f"{self.base_url}/rest/api/content/{page_id}"
            f"?expand=body.storage,children.page,space,ancestors,version"
        )
        resp = self.get(url)
        if resp:
            try:
                return resp.json()
            except Exception as e:
                self.logger.error(f"Failed to parse REST response: {e}")
        return None

    def get_child_pages_rest(self, page_id: str) -> list[dict]:
        """Get child pages via REST API with pagination."""
        children = []
        start = 0
        limit = 50
        while True:
            url = (
                f"{self.base_url}/rest/api/content/{page_id}"
                f"/child/page?start={start}&limit={limit}"
                f"&expand=space,version"
            )
            resp = self.get(url)
            if not resp:
                break
            try:
                data = resp.json()
            except Exception:
                break
            results = data.get("results", [])
            children.extend(results)
            if len(results) < limit:
                break
            start += limit
        return children


# ─── HTML to Markdown Converter ──────────────────────────────────────────────


class ConfluenceToMarkdown:
    """Converts Confluence HTML content to Markdown with link handling."""

    def __init__(
            self,
            base_url: str,
            registry: LinkRegistry,
            current_page_url: str,
            logger_inst,
    ):
        self.base_url = base_url
        self.registry = registry
        self.current_page_url = current_page_url
        self.logger = logger_inst

    def convert(self, html_content: str, page_title: str = "") -> str:
        if not html_content.strip():
            return f"# {page_title}\n\n*Page has no content.*\n"

        soup = BeautifulSoup(html_content, "lxml")

        # Extract and register all links before conversion
        self._extract_links(soup)

        # Process Confluence-specific macros
        self._process_confluence_macros(soup)

        # Convert to markdown
        try:
            markdown_content = md(
                str(soup),
                heading_style="atx",
                bullets="-",
                code_language_callback=self._detect_code_language,
                strip=["script", "style"],
            )
        except Exception as e:
            self.logger.warning(f"markdownify failed, using fallback: {e}")
            markdown_content = soup.get_text(separator="\n\n")

        # Post-process
        markdown_content = self._post_process_markdown(markdown_content)

        header = f"# {page_title}\n\n" if page_title else ""
        header += f"<!-- Source: {self.current_page_url} -->\n\n"

        return header + markdown_content

    def _extract_links(self, soup: BeautifulSoup):
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get_text(strip=True)
            full_url = urljoin(self.base_url + "/", href)
            parsed = urlparse(full_url)
            base_parsed = urlparse(self.base_url)

            is_external = parsed.netloc != base_parsed.netloc

            self.registry.register_link(
                url=full_url,
                title=title,
                is_external=is_external,
                found_on=self.current_page_url,
            )

            # Rewrite link to local file if already parsed
            local_file = self.registry.get_local_file_for_url(full_url)
            if local_file:
                a_tag["href"] = local_file

    def _process_confluence_macros(self, soup: BeautifulSoup):
        # Handle structured macros
        for macro in soup.find_all("ac:structured-macro"):
            macro_name = macro.get("ac:name", "")

            if macro_name == "code":
                lang_param = macro.find("ac:parameter", {"ac:name": "language"})
                lang = lang_param.get_text(strip=True) if lang_param else ""
                body = macro.find("ac:plain-text-body")
                code_text = body.get_text() if body else ""
                new_tag = soup.new_tag("pre")
                code_tag = soup.new_tag("code", attrs={"class": f"language-{lang}"})
                code_tag.string = code_text
                new_tag.append(code_tag)
                macro.replace_with(new_tag)

            elif macro_name in ("info", "note", "warning", "tip"):
                icons = {"info": "ℹ️", "note": "📝", "warning": "⚠️", "tip": "💡"}
                body = macro.find("ac:rich-text-body")
                content = body.decode_contents() if body else ""
                blockquote = soup.new_tag("blockquote")
                prefix = soup.new_tag("strong")
                prefix.string = f"{icons.get(macro_name, '')} {macro_name.upper()}: "
                blockquote.append(prefix)
                blockquote.append(BeautifulSoup(content, "lxml"))
                macro.replace_with(blockquote)

            elif macro_name == "toc":
                macro.decompose()

            else:
                # Generic: extract body content
                body = macro.find("ac:rich-text-body") or macro.find(
                    "ac:plain-text-body"
                )
                if body:
                    div = soup.new_tag("div")
                    div.append(BeautifulSoup(body.decode_contents(), "lxml"))
                    macro.replace_with(div)
                else:
                    macro.decompose()

        # Confluence images
        for img in soup.find_all("ac:image"):
            attachment = img.find("ri:attachment")
            url_elem = img.find("ri:url")
            if attachment:
                filename = attachment.get("ri:filename", "image")
                new_img = soup.new_tag("img", src=filename, alt=filename)
                img.replace_with(new_img)
            elif url_elem:
                src = url_elem.get("ri:value", "")
                new_img = soup.new_tag("img", src=src, alt="image")
                img.replace_with(new_img)
            else:
                img.decompose()

        # Emoticons
        for emoticon in soup.find_all("ac:emoticon"):
            name = emoticon.get("ac:name", "")
            emoji_map = {
                "smile": "😊", "sad": "😢", "cheeky": "😜",
                "laugh": "😂", "wink": "😉", "thumbs-up": "👍",
                "thumbs-down": "👎", "information": "ℹ️",
                "tick": "✅", "cross": "❌", "warning": "⚠️",
            }
            replacement = emoji_map.get(name, f":{name}:")
            emoticon.replace_with(replacement)

    @staticmethod
    def _detect_code_language(el):
        classes = el.get("class", []) if el else []
        for cls in classes:
            if isinstance(cls, str) and cls.startswith("language-"):
                return cls.replace("language-", "")
        return ""

    @staticmethod
    def _post_process_markdown(content: str) -> str:
        content = re.sub(r"\n{4,}", "\n\n\n", content)
        content = re.sub(r"[ \t]+\n", "\n", content)
        content = re.sub(r"^\s+", "", content)
        if not content.endswith("\n"):
            content += "\n"
        return content


# ─── Child Page Detector (XPath-based) ──────────────────────────────────────


class ChildPageDetector:
    """Detects child pages using XPath on the Confluence page HTML."""

    CHILD_PAGE_XPATHS = [
        '//div[contains(@class, "plugin_pagetree")]//a[contains(@class, "plugin_pagetree_children_span")]',
        '//div[@id="children-section"]//a',
        '//div[contains(@class, "children-section")]//a',
        '//div[@id="page-children"]//a',
        '//div[contains(@class, "childpages-macro")]//a',
        '//div[contains(@class, "ia-secondary-container")]//a[contains(@class, "content-type-page")]',
        '//div[contains(@class, "pageSection")][.//strong[contains(text(), "Child")]]//a',
        '//ul[contains(@class, "child-pages")]//a',
    ]

    def __init__(self, base_url: str, logger_inst):
        self.base_url = base_url
        self.logger = logger_inst

    def find_child_links_from_html(self, html_content: str) -> list[dict]:
        children = []
        seen_urls = set()

        try:
            tree = etree.HTML(html_content)
        except Exception as e:
            self.logger.error(f"Failed to parse HTML with lxml: {e}")
            return children

        for xpath_expr in self.CHILD_PAGE_XPATHS:
            try:
                elements = tree.xpath(xpath_expr)
                for elem in elements:
                    href = elem.get("href", "")
                    title = elem.text or elem.get("title", "")
                    if not title:
                        title = "".join(elem.itertext()).strip()
                    if not href:
                        continue

                    full_url = urljoin(self.base_url + "/", href)

                    if not self._is_confluence_page_url(full_url):
                        continue

                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        children.append({"url": full_url, "title": title.strip()})
            except etree.XPathError as e:
                self.logger.debug(f"XPath error for '{xpath_expr}': {e}")

        self.logger.debug(f"Found {len(children)} child links via XPath in HTML")
        return children

    def _is_confluence_page_url(self, url: str) -> bool:
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)
        if parsed.netloc != base_parsed.netloc:
            return False
        path = parsed.path
        page_patterns = [
            r"/wiki/spaces/\w+/pages/\d+",
            r"/display/\w+/.+",
            r"/pages/viewpage\.action",
            r"/wiki/display/\w+/.+",
        ]
        return any(re.search(p, path) for p in page_patterns)


# ─── URL / Page ID Utilities ────────────────────────────────────────────────


def extract_page_id_from_url(url: str) -> Optional[str]:
    match = re.search(r"/pages/(\d+)", url)
    if match:
        return match.group(1)
    match = re.search(r"pageId=(\d+)", url)
    if match:
        return match.group(1)
    return None


def extract_space_key_from_url(url: str) -> Optional[str]:
    match = re.search(r"/spaces/(\w+)/", url)
    if match:
        return match.group(1)
    match = re.search(r"/display/(\w+)/", url)
    if match:
        return match.group(1)
    return None


def make_safe_name(title: str, page_id: str = "") -> str:
    """Create a safe name for folder/file (without extension)."""
    safe = re.sub(r'[<>:"/\\|?*]', "_", title)
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("_.")
    if not safe:
        safe = (
            f"page_{page_id}"
            if page_id
            else hashlib.md5(title.encode()).hexdigest()[:12]
        )
    if len(safe) > 100:
        safe = safe[:100]
    return safe


def make_safe_filename(title: str, page_id: str = "") -> str:
    """Create a safe filename with .md extension."""
    return make_safe_name(title, page_id) + ".md"


def build_page_url(base_url: str, page_id: str) -> str:
    return f"{base_url}/pages/viewpage.action?pageId={page_id}"


# ─── Main Parser ────────────────────────────────────────────────────────────


class ConfluenceParser:
    """Main parser that orchestrates crawling and conversion."""

    def __init__(self, config: dict):
        self.base_url = config["confluence_base_url"].rstrip("/")
        self.start_url = config["start_page_url"]
        self.max_depth = int(config["max_depth"])
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = float(config.get("delay_between_requests", 1))

        self.logger = setup_logging(config.get("log_level", "INFO"))

        self.session = ConfluenceSession(
            self.base_url,
            config["username"],
            config["password"],
            self.logger,
        )
        self.registry = LinkRegistry(
            str(self.output_dir / LINKS_FILE), self.logger
        )
        self.child_detector = ChildPageDetector(self.base_url, self.logger)

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("Confluence Wiki Parser Starting")
        self.logger.info(f"  Base URL:   {self.base_url}")
        self.logger.info(f"  Start URL:  {self.start_url}")
        self.logger.info(f"  Max Depth:  {self.max_depth}")
        self.logger.info(f"  Output Dir: {self.output_dir}")
        self.logger.info("=" * 60)

        if not self.session.login():
            self.logger.error("Authentication failed. Exiting.")
            return False

        self._crawl_page(self.start_url, depth=0, parent_url="", parent_folder=None)

        self.registry.save()
        self._create_combined_output()
        self._print_summary()
        return True

    def _create_combined_output(self):
        """
        Create a single combined markdown file from all parsed pages.
        Pages are organized in hierarchical order with no external links.
        """
        self.logger.info("Creating combined output file...")

        combined_file = self.output_dir / "combined.md"

        # Build hierarchy from parsed pages
        pages_by_url = {p.url: p for p in self.registry.parsed_pages.values()}

        # Find root page (the start page)
        root_page = pages_by_url.get(self.start_url)
        if not root_page:
            self.logger.warning("Could not find root page for combined output")
            return

        def build_hierarchy(page: PageInfo, indent_level: int = 0) -> list[str]:
            """Recursively build markdown content from page hierarchy."""
            lines = []

            # Add page separator (except for root)
            if indent_level > 0:
                lines.append("\n" + "=" * 80 + "\n")

            # Add page title with hierarchy indicator
            title_prefix = "  " * indent_level
            lines.append(f"{title_prefix}# {page.title}\n")
            lines.append(f"*Source: {page.url}*\n")

            # Read the markdown file content
            try:
                file_path = self.output_dir / page.local_file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Remove the title line (already added) and source comment
                content_lines = content.split("\n")
                filtered_lines = []
                skip_next = False
                for i, line in enumerate(content_lines):
                    # Skip title line
                    if i == 0 and line.startswith("#"):
                        continue
                    # Skip source comment
                    if line.startswith("<!-- Source:"):
                        skip_next = True
                        continue
                    if skip_next and line.strip() == "":
                        skip_next = False
                        continue
                    # Skip "Child Pages" section header and its content
                    if line.strip() == "## Child Pages":
                        break
                    filtered_lines.append(line)

                content = "\n".join(filtered_lines)

                # Remove link footer section
                if "---\n\n## Links Found on This Page" in content:
                    content = content.split("---\n\n## Links Found on This Page")[0]

                lines.append(content)

            except Exception as e:
                self.logger.warning(f"Could not read {page.local_file}: {e}")
                lines.append(f"*Content not available*\n")

            # Recursively add child pages
            for child_url in page.children_urls:
                if child_url in pages_by_url:
                    child_page = pages_by_url[child_url]
                    lines.extend(build_hierarchy(child_page, indent_level + 1))

            return lines

        # Build combined content
        combined_lines = [
            "# Combined Confluence Export\n",
            f"Generated from: {self.start_url}\n",
            f"Base URL: {self.base_url}\n",
            "=" * 80 + "\n",
        ]
        combined_lines.extend(build_hierarchy(root_page))

        # Write combined file
        with open(combined_file, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_lines))

        self.logger.info(f"Combined output saved to: {combined_file}")

    def _crawl_page(
        self,
        page_url: str,
        depth: int,
        parent_url: str,
        parent_folder: Optional[Path] = None,
    ):
        """
        Crawl and parse a Confluence page.

        Args:
            page_url: URL of the page to parse
            depth: Current depth in the hierarchy
            parent_url: URL of the parent page
            parent_folder: Path to the parent folder (for nested structure)
        """
        if depth > self.max_depth:
            self.logger.info(
                f"{'  ' * depth}Max depth {self.max_depth} reached, skipping: {page_url}"
            )
            return

        if self.registry.is_parsed(page_url):
            self.logger.info(f"{'  ' * depth}Already parsed, skipping: {page_url}")
            return

        self.logger.info(f"{'  ' * depth}[Depth {depth}] Parsing: {page_url}")

        page_id = extract_page_id_from_url(page_url)
        space_key = extract_space_key_from_url(page_url) or ""

        page_data = None
        html_content = ""
        page_title = ""
        child_pages = []

        # ── Strategy 1: REST API (if we have a page ID) ──
        if page_id:
            self.logger.debug(f"Trying REST API for page ID: {page_id}")
            page_data = self.session.get_page_by_rest_api(page_id)

        if page_data:
            page_title = page_data.get("title", f"Page_{page_id}")
            html_content = (
                page_data.get("body", {}).get("storage", {}).get("value", "")
            )
            space_key = page_data.get("space", {}).get("key", space_key)

            # Get children via REST API (these are TRUE children, not links)
            rest_children = self.session.get_child_pages_rest(page_id)
            self.logger.info(
                f"{'  ' * depth}  REST API: found {len(rest_children)} child pages"
            )
            for child in rest_children:
                child_id = child.get("id", "")
                child_title = child.get("title", "")
                child_space = child.get("space", {}).get("key", space_key)
                child_url = (
                    f"{self.base_url}/spaces/{child_space}/pages/{child_id}/"
                    f"{requests.utils.quote(child_title)}"
                )
                child_pages.append(
                    {"url": child_url, "title": child_title, "id": child_id}
                )

        else:
            # ── Strategy 2: Fetch raw HTML + XPath ──
            self.logger.info(
                f"{'  ' * depth}  No REST data, fetching HTML directly..."
            )
            resp = self.session.get(page_url)
            if not resp:
                self.logger.error(f"Failed to fetch page: {page_url}")
                return

            full_html = resp.text

            # Check for SSO/auth redirect page
            if self.session._is_sso_redirect_page(full_html):
                self.logger.error(
                    f"Authentication required! Got SSO redirect page instead of content.\n"
                    f"Please log in to Confluence in your browser first, then run the script again.\n"
                    f"URL: {page_url}"
                )
                return

            # Extract title via XPath
            try:
                tree = etree.HTML(full_html)
                title_xpaths = [
                    '//meta[@name="ajs-page-title"]/@content',
                    '//h1[@id="title-text"]/a/text()',
                    '//h1[@id="title-text"]/text()',
                    '//div[@id="title-heading"]//text()',
                    '//title/text()',
                ]
                for xp in title_xpaths:
                    results = tree.xpath(xp)
                    if results:
                        page_title = "".join(results).strip()
                        # Clean up title (remove " - Confluence" suffix etc.)
                        page_title = re.sub(
                            r"\s*[-–]\s*Confluence\s*$", "", page_title
                        )
                        if page_title:
                            break
            except Exception as e:
                self.logger.warning(f"XPath title extraction failed: {e}")

            if not page_title:
                page_title = f"Page_{page_id or 'unknown'}"

            # Extract main content via XPath
            content_xpaths = [
                '//div[@id="main-content"]',
                '//div[contains(@class, "wiki-content")]',
                '//div[@id="content-body"]',
                '//div[contains(@class, "page-content")]',
                '//section[@id="content"]//div[contains(@class, "aui-item")]',
                "//article",
            ]
            try:
                tree = etree.HTML(full_html)
                for xp in content_xpaths:
                    content_elements = tree.xpath(xp)
                    if content_elements:
                        html_content = etree.tostring(
                            content_elements[0],
                            encoding="unicode",
                            method="html",
                        )
                        self.logger.debug(
                            f"Content extracted via XPath: {xp}"
                        )
                        break
                if not html_content:
                    self.logger.warning("No content found via XPath, using full HTML")
                    html_content = full_html
            except Exception as e:
                self.logger.warning(f"Content extraction failed: {e}")
                html_content = full_html

            # Try to get page ID from HTML meta tags
            if not page_id:
                try:
                    tree = etree.HTML(full_html)
                    id_xpaths = [
                        '//meta[@name="ajs-page-id"]/@content',
                        '//meta[@name="confluence-page-id"]/@content',
                    ]
                    for xp in id_xpaths:
                        results = tree.xpath(xp)
                        if results:
                            page_id = results[0]
                            self.logger.debug(
                                f"Found page ID from HTML meta: {page_id}"
                            )
                            break
                except Exception:
                    pass

            # Find child pages from HTML via XPath
            child_pages_raw = self.child_detector.find_child_links_from_html(
                full_html
            )
            child_pages = [
                {
                    "url": c["url"],
                    "title": c["title"],
                    "id": extract_page_id_from_url(c["url"]) or "",
                }
                for c in child_pages_raw
            ]

            # Also try REST API for children if we discovered a page ID
            if page_id and not child_pages:
                self.logger.debug(
                    f"Trying REST API for children with discovered page ID: {page_id}"
                )
                rest_children = self.session.get_child_pages_rest(page_id)
                for child in rest_children:
                    child_id = child.get("id", "")
                    child_title = child.get("title", "")
                    child_url = build_page_url(self.base_url, child_id)
                    child_pages.append(
                        {"url": child_url, "title": child_title, "id": child_id}
                    )

        if not page_id:
            page_id = hashlib.md5(page_url.encode()).hexdigest()[:12]

        self.logger.info(
            f"{'  ' * depth}  Title: {page_title} | "
            f"Children: {len(child_pages)}"
        )

        # ── Convert HTML → Markdown ──
        converter = ConfluenceToMarkdown(
            self.base_url, self.registry, page_url, self.logger
        )
        markdown_content = converter.convert(html_content, page_title)

        # Add child pages section
        if child_pages:
            markdown_content += "\n\n---\n\n## Child Pages\n\n"
            for child in child_pages:
                markdown_content += f"- [{child['title']}]({child['url']})\n"

        # Add link footer
        markdown_content += self._build_link_footer(page_url)

        # ── Save file ──
        # Create nested folder structure:
        # - Each page's .md file is saved in its parent's folder
        # - A subfolder with the same name is created for its children
        # Example structure:
        #   Home.md
        #   Home/
        #     Getting Started.md
        #     Getting Started/
        #       Installation.md
        safe_name = make_safe_name(page_title, page_id)
        filename = safe_name + ".md"

        # Determine where to save this page's .md file
        if parent_folder is None:
            # Root level: save directly in output_dir
            save_folder = self.output_dir
        else:
            # Nested: save in parent's folder
            save_folder = parent_folder

        # Create the save folder if needed
        save_folder.mkdir(parents=True, exist_ok=True)
        filepath = save_folder / filename

        # Handle filename collisions
        counter = 1
        original_filepath = filepath
        while filepath.exists():
            stem = original_filepath.stem
            filepath = original_filepath.with_name(f"{stem}_{counter}.md")
            counter += 1

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        relative_path = str(filepath.relative_to(self.output_dir))
        self.logger.info(f"{'  ' * depth}  → Saved: {relative_path}")

        # Create a subfolder for this page's children (same name as the .md file)
        # This folder will contain the children's .md files and their subfolders
        # Only create if there are children
        children_folder = None
        if child_pages:
            children_folder = save_folder / safe_name
            children_folder.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"{'  ' * depth}  Created children folder: {children_folder}")

        # ── Register in registry ──
        page_info = PageInfo(
            url=page_url,
            title=page_title,
            page_id=page_id,
            space_key=space_key,
            depth=depth,
            parent_url=parent_url,
            children_urls=[c["url"] for c in child_pages],
            local_file=relative_path,
        )
        self.registry.register_parsed_page(page_info)
        self.registry.register_link(
            url=page_url,
            title=page_title,
            parsed=True,
            is_child=(depth > 0),
            local_file=relative_path,
        )

        for child in child_pages:
            self.registry.register_link(
                url=child["url"],
                title=child["title"],
                is_child=True,
                found_on=page_url,
            )

        # Save after each page for resume capability
        self.registry.save()

        # ── Recurse into child pages ──
        # Pass the children folder as parent for children (only if there are children)
        if children_folder and child_pages:
            for child in child_pages:
                time.sleep(self.delay)
                self._crawl_page(
                    child["url"],
                    depth + 1,
                    parent_url=page_url,
                    parent_folder=children_folder,
                )

    def _build_link_footer(self, page_url: str) -> str:
        footer = "\n\n---\n\n## Links Found on This Page\n\n"

        internal_parsed = []
        internal_unparsed = []
        external = []

        for url, entry in self.registry.links.items():
            if page_url in entry.found_on_pages:
                if entry.is_external:
                    external.append(entry)
                elif entry.parsed:
                    internal_parsed.append(entry)
                else:
                    internal_unparsed.append(entry)

        if internal_parsed:
            footer += "### Internal (Parsed)\n\n"
            for entry in internal_parsed:
                link_target = entry.local_file or entry.url
                footer += f"- [{entry.title or entry.url}]({link_target})\n"

        if internal_unparsed:
            footer += "\n### Internal (Not Yet Parsed)\n\n"
            for entry in internal_unparsed:
                footer += f"- [{entry.title or entry.url}]({entry.url})\n"

        if external:
            footer += "\n### External Links\n\n"
            for entry in external:
                footer += f"- [{entry.title or entry.url}]({entry.url})\n"

        if not (internal_parsed or internal_unparsed or external):
            footer += "*No links found.*\n"

        return footer

    def _print_summary(self):
        total_parsed = len(self.registry.parsed_pages)
        total_links = len(self.registry.links)
        unparsed = self.registry.get_unparsed_internal_links()
        external = [e for e in self.registry.links.values() if e.is_external]

        self.logger.info("\n" + "=" * 60)
        self.logger.info("PARSING COMPLETE — SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"  Pages parsed:              {total_parsed}")
        self.logger.info(f"  Total links discovered:    {total_links}")
        self.logger.info(f"  Internal unparsed links:   {len(unparsed)}")
        self.logger.info(f"  External links:            {len(external)}")
        self.logger.info(f"  Output directory:          {self.output_dir}")
        self.logger.info(
            f"  Link registry:             {self.output_dir / LINKS_FILE}"
        )

        if unparsed:
            self.logger.info(
                "\n  Unparsed internal links (can continue parsing):"
            )
            for entry in unparsed[:20]:
                self.logger.info(
                    f"    - {entry.title or 'Untitled'}: {entry.url}"
                )
            if len(unparsed) > 20:
                self.logger.info(f"    ... and {len(unparsed) - 20} more")

        self.logger.info("=" * 60)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Confluence → Markdown Parser")
    print("=" * 60)
    print()

    config = load_config()
    parser = ConfluenceParser(config)
    success = parser.run()

    if not success:
        print("\nParser failed. Check the log for details.")
        raise SystemExit(1)