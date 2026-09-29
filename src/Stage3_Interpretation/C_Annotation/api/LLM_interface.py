#%%
import anthropic
from pydantic import ValidationError
from pydantic import BaseModel


class ProgramAnnotation(BaseModel):
    program_name: str
    top_pathways: list[str]
    confidence: float
    rationale: str

class claude_api:
    def __init__(self, api_key, model):
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=600)
        self.model = model

    def get_completion(self, system_prompt, prompt, effort="medium", max_tokens=4000):
        try: 
            msg = self.client.messages.create(
                model=self.model, 
                max_tokens=max_tokens, 
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                output_config = {"effort": effort}, # "low" | "medium" | "high" | "xhigh" | "max"
            )
            if msg.stop_reason == "max_tokens":
                print("Warning: answer was cut off")
            
            return "".join(b.text for b in msg.content if b.type == "text")
        
        except anthropic.APIError as e:
            print(e)
            return None

    def get_json(self, system_prompt, prompt, schema_model, effort="medium", max_tokens=4000):
        try: 
            msg = self.client.messages.parse(
                model=self.model, 
                max_tokens=max_tokens, 
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema_model,
                output_config = {"effort": effort}, # "low" | "medium" | "high" | "xhigh" | "max"
            )

        except ValidationError as e:
            print("Validation failed:", e)
            return None

        except anthropic.APIError as e:
            print("API error:", e)
            return None

        if msg.stop_reason == "max_tokens":
            print("Warning: answer was cut off")
            return None

        if msg.stop_reason == "refusal":
            print("Claude refused this request")
            return None


        return msg.parsed_output  

# %%
