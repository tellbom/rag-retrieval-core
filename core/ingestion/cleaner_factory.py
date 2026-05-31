"""
core/ingestion/cleaner_factory.py

CleanerFactory: builds a Cleaner from a CleaningProfile or config dict.

Design
------
- No business-type logic.  The factory is profile-in → Cleaner-out.
- Profile patterns are compiled at factory call time (fail-fast on bad regex).
- Business adapters that need custom boilerplate rules load a CleaningProfile
  from their own config files and pass it here.  The retrieval core never
  hard-codes domain knowledge.

Usage
-----
    # Generic cleaner (no extra patterns, default boilerplate on):
    cleaner = CleanerFactory.generic()

    # From a config dict (e.g. loaded from JSON profile file):
    profile = CleaningProfile.from_dict({
        "boilerplate_patterns": [
            {"pattern": "^Footer.*$", "flags": ["MULTILINE"]},
        ],
        "disable_default_boilerplate": False,
    })
    cleaner = CleanerFactory.from_profile(profile)

    # From raw regex strings (convenience; all share the same flags):
    cleaner = CleanerFactory.from_pattern_strings(
        ["^Footer.*$", "^Advertisement.*$"],
        flags=re.MULTILINE,
    )
"""

from __future__ import annotations

import re

from core.ingestion.cleaner import Cleaner
from core.ingestion.cleaning_profile import CleaningProfile


class CleanerFactory:
    """
    Constructs Cleaner instances from profiles or config dicts.
    All methods are static — the factory has no state.
    """

    @staticmethod
    def generic() -> Cleaner:
        """
        Return a Cleaner with no extra patterns and default boilerplate enabled.
        Suitable as the baseline for any document type.
        """
        return Cleaner(profile=CleaningProfile.empty())

    @staticmethod
    def from_profile(profile: CleaningProfile) -> Cleaner:
        """
        Build a Cleaner from a pre-constructed CleaningProfile.
        Pattern compilation already happened at profile creation time.
        """
        return Cleaner(profile=profile)

    @staticmethod
    def from_dict(config: dict) -> Cleaner:
        """
        Build a Cleaner from a plain dict (e.g. from a JSON profile file).
        Compiles all regex patterns immediately — raises CleaningProfileError
        on any invalid pattern or unknown flag.

        See CleaningProfile.from_dict() for the expected dict shape.
        """
        profile = CleaningProfile.from_dict(config)
        return Cleaner(profile=profile)

    @staticmethod
    def from_pattern_strings(
        patterns: list[str],
        *,
        flags: int = re.MULTILINE,
        disable_default_boilerplate: bool = False,
    ) -> Cleaner:
        """
        Build a Cleaner from a list of raw regex strings.
        All patterns share the same `flags`.
        Raises CleaningProfileError on any invalid regex (fail-fast).
        """
        profile = CleaningProfile.from_pattern_strings(
            patterns,
            flags=flags,
            disable_default_boilerplate=disable_default_boilerplate,
        )
        return Cleaner(profile=profile)
