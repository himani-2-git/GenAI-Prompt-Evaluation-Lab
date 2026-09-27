import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from src.llm_client import GeminiClient

client = GeminiClient()
print(f"Key loaded: {client.is_configured()}")
print(f"Model: {client.model_name}")
print("Sending test prompt...\n")

result = client.generate(
    "Say exactly: 'API connection successful.' and nothing else.",
    temperature=0.0,
    max_tokens=200,
)

print(f"Success:     {result['success']}")
print(f"Response:    {repr(result['text'])}")
print(f"Latency:     {result['latency_ms']:.0f}ms")
print(f"Tokens:      input={result['input_tokens']}, output={result['output_tokens']}, total={result['total_tokens']}")
print(f"Finish:      {result['finish_reason']}")
if not result['success']:
    print(f"Error:       {result['error']}")
