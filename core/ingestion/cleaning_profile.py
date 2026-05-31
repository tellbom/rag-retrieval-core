"""
core/ingestion/cleaning_profile.py

CleaningProfile: a data object that carries externally-configured boilerplate
regex patterns for the Cleaner.

Design
------
- All regex strings are **compiled at profile construction time**.
  Invalid patterns raise CleaningProfileError immediately (fail-fast),
  not at document-processing time.

- A profile has no knowledge of business types.  The caller (adapter,
  ingestion API handler, or test) decides which profile to pass to
  CleanerFactory.from_profile().

- Profiles can be loaded from:
    1. A plain Python dict (e.g. parsed from JSON / YAML config file).
    2. A list of raw regex strings.
    3. Directly constructed in code (for tests or custom adapters).

- The schema for a profile dict (e.g. read from a per-tenant JSON):
    {
      "boilerplate_patterns": [
          {"pattern": "^Footer.*$",      "flags": ["MULTILINE"]},
          {"pattern": "^广告.*$",         "flags": ["MULTILINE", "IGNORECASE"]},
          ...
      ],
      "disable_default_boilerplate": false   // optional, default false
    }

Public API
----------
    profile = CleaningProfile.from_dict(config_dict)
    profile = CleaningProfile.from_pattern_strings(["^Footer.*$"], flags=re.MULTILINE)
    profile = CleaningProfile()   # empty profile — no extra patterns

    cleaner = CleanerFactory.from_profile(profile)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FLAG_MAP: dict[str, int] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE":  re.MULTILINE,
    "DOTALL":     re.DOTALL,
    "VERBOSE":    re.VERBOSE,
    "ASCII":      re.ASCII,
    "UNICODE":    re.UNICODE,
}


class CleaningProfileError(ValueError):
    """Raised when a CleaningProfile cannot be constructed (bad regex, bad flag)."""


@dataclass
class CleaningProfile:
    """
    Holds compiled extra-boilerplate patterns for a Cleaner instance.

    Attributes
    ----------
    extra_boilerplate_patterns:
        Compiled regex patterns prepended to the default boilerplate list.
        Empty by default — core cleaning only.
    disable_default_boilerplate:
        If True, the built-in default boilerplate patterns in rules.py are
        suppressed.  Useful when a profile wants full control over what
        counts as boilerplate.  Default False.
    """
    extra_boilerplate_patterns: list[re.Pattern] = field(default_factory=list)
    disable_default_boilerplate: bool = False

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "CleaningProfile":
        """
        Build a CleaningProfile from a plain dict (e.g. parsed from JSON).

        Expected shape::

            {
              "boilerplate_patterns": [
                  {"pattern": "^Footer.*$", "flags": ["MULTILINE"]},
                  ...
              ],
              "disable_default_boilerplate": false
            }

        Raises CleaningProfileError on any invalid regex or unknown flag.
        All patterns are compiled immediately (fail-fast).
        """
        raw_patterns: list[dict] = data.get("boilerplate_patterns", [])
        disable_defaults: bool = bool(data.get("disable_default_boilerplate", False))

        compiled: list[re.Pattern] = []
        for i, entry in enumerate(raw_patterns):
            if not isinstance(entry, dict) or "pattern" not in entry:
                raise CleaningProfileError(
                    f"boilerplate_patterns[{i}] must be a dict with a 'pattern' key, "
                    f"got: {entry!r}"
                )
            pattern_str: str = entry["pattern"]
            flag_names: list[str] = entry.get("flags", ["MULTILINE"])

            flags = cls._parse_flags(flag_names, context=f"patterns[{i}]")
            compiled.append(cls._compile(pattern_str, flags, context=f"patterns[{i}]"))

        return cls(
            extra_boilerplate_patterns=compiled,
            disable_default_boilerplate=disable_defaults,
        )

    @classmethod
    def from_pattern_strings(
        cls,
        patterns: list[str],
        *,
        flags: int = re.MULTILINE,
        disable_default_boilerplate: bool = False,
    ) -> "CleaningProfile":
        """
        Build a CleaningProfile from a list of raw regex strings, all sharing
        the same `flags`.  Compiled immediately — fail-fast on bad regex.
        """
        compiled = [
            cls._compile(p, flags, context=f"pattern[{i}]")
            for i, p in enumerate(patterns)
        ]
        return cls(
            extra_boilerplate_patterns=compiled,
            disable_default_boilerplate=disable_default_boilerplate,
        )

    @classmethod
    def empty(cls) -> "CleaningProfile":
        """Profile with no extra patterns and defaults enabled."""
        return cls()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_flags(flag_names: list[str], context: str) -> int:
        result = 0
        for name in flag_names:
            upper = name.upper()
            if upper not in _FLAG_MAP:
                raise CleaningProfileError(
                    f"{context}: unknown regex flag {name!r}. "
                    f"Allowed: {list(_FLAG_MAP)}"
                )
            result |= _FLAG_MAP[upper]
        return result

    @staticmethod
    def _compile(pattern: str, flags: int, context: str) -> re.Pattern:
        try:
            return re.compile(pattern, flags)
        except re.error as exc:
            raise CleaningProfileError(
                f"{context}: invalid regex {pattern!r}: {exc}"
            ) from exc
