#%%
import anthropic
from pydantic import ValidationError
from pydantic import BaseModel

from budget import claude_cost, claude_worst_case


class ProgramAnnotation(BaseModel):
    program_name: str
    top_pathways: list[str]
    confidence: float
    rationale: str

class claude_api:
    def __init__(self, api_key, model):
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3, timeout=600)
        self.model = model
        # optional spend cap (api/budget.attach): budget.ProgramBudget, step name, USD left for later steps
        self.budget, self.budget_step, self.budget_reserve = None, "", 0.0
        self.last_error = None   # why the latest get_json returned None (for the step's run record)

    def _fits(self, system_prompt, prompt, max_tokens):
        """False (and recorded as skipped) when the call's worst case does not fit the program's budget."""
        if self.budget is None:
            return True
        estimate = claude_worst_case(self.model, len(system_prompt) + len(prompt), max_tokens)
        if self.budget.allows(estimate, self.budget_reserve):
            return True
        self.budget.skip(self.budget_step, f"{self.model} call", estimate)
        return False

    def _charge(self, msg):
        if self.budget is not None and getattr(msg, "usage", None) is not None:
            u = msg.usage
            self.budget.charge(self.budget_step, claude_cost(self.model, u), f"{self.model} call",
                               input_tokens=u.input_tokens, output_tokens=u.output_tokens)

    def get_completion(self, system_prompt, prompt, effort="medium", max_tokens=4000):
        if not self._fits(system_prompt, prompt, max_tokens):
            return None
        try: 
            msg = self.client.messages.create(
                model=self.model, 
                max_tokens=max_tokens, 
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                output_config = {"effort": effort}, # "low" | "medium" | "high" | "xhigh" | "max"
            )
            self._charge(msg)
            if msg.stop_reason == "max_tokens":
                print("Warning: answer was cut off")
            
            return "".join(b.text for b in msg.content if b.type == "text")
        
        except anthropic.APIError as e:
            print(e)
            return None

    def get_json(self, system_prompt, prompt, schema_model, effort="medium", max_tokens=4000):
        self.last_error = None
        if not self._fits(system_prompt, prompt, max_tokens):
            self.last_error = "skipped: worst-case cost does not fit the program budget"
            return None
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
            self.last_error = f"answer does not match the schema: {e}"[:500]
            return None

        except anthropic.APIError as e:
            print("API error:", e)
            self.last_error = f"API error: {e}"[:500]
            return None

        self._charge(msg)   # billed even when cut off or refused
        if msg.stop_reason == "max_tokens":
            print("Warning: answer was cut off")
            self.last_error = "answer cut off at max_tokens"
            return None

        if msg.stop_reason == "refusal":
            print("Claude refused this request")
            self.last_error = "refused"
            return None


        return msg.parsed_output  

# %%
