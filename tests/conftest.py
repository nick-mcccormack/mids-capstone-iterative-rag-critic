"""Pytest configuration and Hypothesis settings for LLM Critic Enhancement tests.

This module configures the Hypothesis property-based testing framework with
appropriate settings for the critic system tests.
"""

from hypothesis import settings, Verbosity

# Configure Hypothesis settings profile for critic system tests
settings.register_profile(
    "critic_tests",
    max_examples=100,  # Minimum 100 iterations per property test as per design
    verbosity=Verbosity.normal,
    deadline=None,  # Disable deadline for tests that may involve LLM calls
)

# Activate the profile
settings.load_profile("critic_tests")
