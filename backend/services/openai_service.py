import base64
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_thumbnails(prompt: str, style_prompt: str, headshot_url: str) -> bytes:
    """
    Use the Response API with the gpt-image-2 as a built-in image generation tool.
    Pass the headshot_url directly as a input image.
    Returns raw PNG bytes.
    """

    full_prompt=(
        f"{style_prompt}\n\n"
        f"User Request:{prompt}\n\n"
        "IMPORTANT: The generated thumbanil MUST prominently feature the person"
        "shoewn in the provided refernce headshot photo, keep thier likeness accurate "
    )

    response = await client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content":[
                    {"type": "input_image", "url": headshot_url},
                    {"type": "text", "text": full_prompt}
                ]
            }
        ],
        tools= [
            {
                "type": "image_generation",
                "model": "gpt-image-2",
                "size": "1536x1024",
                "quality": "high",
                "output_format": "png",
            },
        ],
        )

    for item in response.output:
        if item.type == "image_generation_call" and item.result:
            return base64.b64decode(item.result)

    raise RuntimeError("No image generation result found in response")
