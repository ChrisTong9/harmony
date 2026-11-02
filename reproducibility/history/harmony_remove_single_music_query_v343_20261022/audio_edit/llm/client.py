import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class LLMDecision:
    script_name: str
    params: dict[str,Any]


class LLMClient:
    def __init__(self):
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path)
        self.backend = os.environ.get("LLM_BACKEND", "").strip().lower()

        if self.backend == "groq":
            self.client = OpenAI(
                base_url=os.environ["GROQ_BASE_URL"],
                api_key=os.environ["GROQ_API_KEY"],
            )
            self.model = os.environ["GROQ_MODEL"]
        elif self.backend =="openai":
            self.client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
            )
            self.model = os.environ["OPENAI_MODEL"]


            create = self.client.chat.completions.create

            def create_json(*args, **kwargs):
                kwargs.setdefault('response_format', {'type': 'json_object'})
                response = create(*args, **kwargs)
                if not kwargs.get('stream'):
                    for choice in response.choices:
                        if choice.finish_reason in ('length', 'content_filter'):
                            raise ValueError(f'Incomplete planner response: {choice.finish_reason}')
                        if getattr(choice.message, 'refusal', None):
                            raise ValueError('Planner response was refused; no executable JSON plan')
                return response

            self.client.chat.completions.create = create_json
        else:
            raise ValueError(
                "LLM_BACKEND must be either 'openai' or 'groq'; "
                f"got {self.backend!r}."
            )

    def choose_script(self,user_prompt:str, script_catalog:dict[str,Any]) -> LLMDecision:
        system_prompt = self._build_system_prompt()
        few_shot_examples = self._build_few_shot_examples()

        user_payload = {
            "user_prompt": user_prompt,
            "available_scripts": script_catalog,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            *few_shot_examples,
            {"role": "user", "content": json.dumps(user_payload,ensure_ascii=False)},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        raw_text = response.choices[0].message.content.strip()

        data = self._parse_json(raw_text)

        self._validate_decision(data,script_catalog)

        return LLMDecision(
            script_name=data["script_name"],
            params=data.get("params",{})
        )

    def _build_system_prompt(self) -> str:
        return """
You are the brain of an audio editing agent.

Your job is to select exactly one script from the available script catalog.

You must return ONLY valid JSON.
Do not use markdown.
Do not explain your reasoning.

Required JSON format:
{
  "script_name": "one_script_name_from_catalog",
  "params": {
    "key": "value"
  }
}

Important rules:
1. Choose only one script_name from the available_scripts.
2. Never invent script names.
3. Keep params simple and explicit. Do not paraphrase away timestamps,
   temporal relations, occurrence words, or numeric counts; the complete user
   instruction remains authoritative for the selected specialist.
4. If the request asks to add a sound, choose add_sound.
5. If the request asks to replace one sound with another, choose replace.
6. If the request asks to remove, isolate, or separate sounds, choose remove_sound.
7. If the request explicitly asks to change volume/loudness/gain, choose
   volume_change. Semantic changes such as "more intense" are not automatically
   volume changes.
8. If uncertain, choose the closest script and include the uncertainty in params.

Current task:
Convert the user's natural language audio editing request into a structured script invocation.
"""

    def _build_few_shot_examples(self) -> list[dict[str, str]]:
        """
        Few-shot examples live here.

        For now, keep a few simple examples.
        Later, you can slowly add better examples from real user cases.

        Format:
        user message: natural language request + available script catalog
        assistant message: pure JSON decision
        """

        examples = []


        examples.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": "add dog barking to the background music",
                        "available_scripts": {
                            "add_sound": {
                                "description": "Add a target sound to the input audio."
                            },
                            "replace": {
                                "description": "Replace one sound with another sound."
                            },
                            "remove_sound": {
                                "description": "Remove sound from the input or keep the rest."
                            },
                            "volume_change": {
                                "description": "Increase or decrease the volume of a target sound."
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )

        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "script_name": "add_sound",
                        "params": {
                            "target_sound": "dog barking",
                            "placement": "background",
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )


        examples.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": "replace the piano with electric guitar",
                        "available_scripts": {
                            "add_sound": {
                                "description": "Add a target sound to the input audio."
                            },
                            "replace": {
                                "description": "Replace one sound with another sound."
                            },
                            "remove_sound": {
                                "description": "Remove sound from the input or keep the rest."
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )

        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "script_name": "replace",
                        "params": {
                            "source_sound": "piano",
                            "target_sound": "electric guitar",
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )


        examples.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": "remove the vocals from this song",
                        "available_scripts": {
                            "add_sound": {
                                "description": "Add a target sound to the input audio."
                            },
                            "replace": {
                                "description": "Replace one sound with another sound."
                            },
                            "remove_sound": {
                                "description": "Remove sound from the input or keep the rest."
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )

        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "script_name": "remove_sound",
                        "params": {
                            "target_sound": "vocals",
                            "operation": "remove",
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )

        examples.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": "increase the sound of dog barking by 10 dB",
                        "available_scripts": {
                            "add_sound": {
                                "description": "Add a target sound to the input audio."
                            },
                            "replace": {
                                "description": "Replace one sound with another sound."
                            },
                            "remove_sound": {
                                "description": "Remove sound from the input or keep the rest."
                            },
                            "volume_change": {
                                "description": "Increase or decrease volume."
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )
        examples.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "script_name": "volume_change",
                        "params": {
                            "target_sound": "dog barking",
                            "gain_db": 10.0,
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )


        return examples

    def _parse_json(self, raw_text:str) -> dict[str,Any]:

        cleaned = raw_text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").strip()

        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                "LLM did not return valid JSON.\n"
                f"Raw output:\n{raw_text}\n"
                f"Cleaned output:\n{cleaned}"
            ) from e


    def _validate_decision(self, data: dict[str, Any], script_catalog: dict[str, Any]) -> None:
        if "script_name" not in data:
            raise ValueError(f"LLM output missing script_name: {data}")

        if data["script_name"] not in script_catalog:
            raise ValueError(
                f"LLM selected unknown script: {data['script_name']}. "
                f"Available scripts: {list(script_catalog.keys())}"
            )

        if "params" in data and not isinstance(data["params"], dict):
            raise ValueError(f"LLM params must be a dict, got: {type(data['params'])}")
