class ScriptRegistry:
    def __init__(self):
        self.scripts = {
            "add_sound": {
                "description": "Add a target sound to the input audio.",
                "module": "audio_edit.user_scripts.add_sound",
            },
            "replace": {
                "description": "Replace one sound with another sound.",
                "module": "audio_edit.user_scripts.replace",
            },
            "remove_sound": {
                "description": "Remove sound from the input.",
                "module": "audio_edit.user_scripts.remove_sound",
            },
            "volume_change": {
                "description": "Increase or decrease volume globally or for a timed event.",
                "module": "audio_edit.user_scripts.volume_change",
            },
        }


    def get(self, script_name: str):
        if script_name not in self.scripts:
            raise ValueError(f"Unknown script: {script_name}")
        return self.scripts[script_name]

    def list_scripts(self):
        return self.scripts
