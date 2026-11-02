from audio_edit.models import AudioEditContext
from audio_edit.scripts.registry import ScriptRegistry
from audio_edit.scripts.runner import ScriptRunner
from audio_edit.agents.music_edit_agent import MusicEditAgent
from audio_edit.llm.client import LLMClient


def main():
    registry = ScriptRegistry()
    runner = ScriptRunner(registry)
    llm = LLMClient()

    agent = MusicEditAgent(
        runner=runner,
        registry=registry,
        llm=llm,
    )

    test_prompts = [
        "replace the dog barking sound with cat meowing sound and also replace the dog with cat in the video",
    ]

    for prompt in test_prompts:


        ctx = AudioEditContext(
            user_prompt=prompt,
            input_audio="./audio_edit/inputs/example1.mp4",
        )

        try:
            result = agent.execute(ctx)


            print(f"Output audio: {result.output_audio}")


        except NotImplementedError as e:

            print(e)


if __name__ == "__main__":
    main()
