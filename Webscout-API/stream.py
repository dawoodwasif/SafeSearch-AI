import asyncio
import json
import aiohttp

async def fastai_stream(user, model="llama3-8b", system="Answer as concisely as possible."):
    env_type = "tp16405b" if "405b" in model else "tp16"
    data = {'body': {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'stream': True, 'model': model}, 'env_type': env_type}
    
    async with aiohttp.ClientSession() as session:
        async with session.post('https://fast.snova.ai/api/completion', headers={'content-type': 'application/json'}, json=data) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data:'):
                    try:
                        data = json.loads(line[len('data: '):])
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", '')
                        if content:
                            yield f"data: {json.dumps({'response': content})}\n\n"
                    except json.JSONDecodeError:
                        if line[len('data: '):] == '[DONE]':
                            break

