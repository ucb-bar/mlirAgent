"""File-based LLM for GEPA -- writes prompts to disk, polls for responses.

GEPA's LM interface is a simple synchronous callable:
``__call__(prompt: str | list[dict]) -> str``

This class writes each prompt as a Markdown file and waits for the user
(or an agent) to create a corresponding ``.response.md`` file.

Usage::

    lm = ManualLM("gepa_prompts")
    response = lm("Write improved code...")  # blocks until response file exists
"""

import os
import time


class ManualLM:
    """File-based LLM for GEPA.

    Writes prompts as ``prompt_NNN.md`` and polls for ``prompt_NNN.response.md``.
    """

    def __init__(self, prompts_dir="gepa_prompts", poll_interval=2.0):
        self.prompts_dir = prompts_dir
        self.poll_interval = poll_interval
        self._counter = 0
        os.makedirs(prompts_dir, exist_ok=True)

    def __call__(self, prompt):
        """Send prompt and block until response file appears."""
        self._counter += 1
        prompt_path = os.path.join(
            self.prompts_dir, f"prompt_{self._counter:03d}.md"
        )
        response_path = os.path.join(
            self.prompts_dir, f"prompt_{self._counter:03d}.response.md"
        )

        with open(prompt_path, "w") as f:
            if isinstance(prompt, str):
                f.write(f"# User\n\n{prompt}\n")
            else:
                # list[dict] format: [{"role": "system", "content": "..."}, ...]
                for msg in prompt:
                    role = msg.get("role", "user").title()
                    content = msg.get("content", "")
                    f.write(f"# {role}\n\n{content}\n\n")

        print(f"  [ManualLM] Prompt written to {prompt_path}")
        print(f"  [ManualLM] Waiting for response at {response_path}...")

        while not os.path.exists(response_path):
            time.sleep(self.poll_interval)

        with open(response_path) as f:
            response = f.read().strip()
        print(f"  [ManualLM] Got response ({len(response)} chars)")
        return response
