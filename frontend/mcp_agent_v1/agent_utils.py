
from google.genai.types import Content, Part

def part_to_dict(p: Part) -> dict:
    if getattr(p, "text", None) is not None:
        return {"text": p.text}

    fc = getattr(p, "function_call", None)
    if fc is not None:
        return {
            "function_call": {
                "name": getattr(fc, "name", None),
                "args": getattr(fc, "args", None),
            }
        }

    fr = getattr(p, "function_response", None)
    if fr is not None:
        resp = getattr(fr, "response", None)
        try:
            json.dumps(resp)
        except Exception:
            resp = str(resp)
        return {
            "function_response": {
                "name": getattr(fr, "name", None),
                "response": resp,
            }
        }

    return {"_unsupported_part": str(p)}


def content_to_dict(c: Content) -> dict:
    return {
        "role": getattr(c, "role", None),
        "parts": [part_to_dict(p) for p in getattr(c, "parts", [])],
    }
