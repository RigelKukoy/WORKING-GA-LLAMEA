# Google Gen AI SDK
pypipypi v1.66.0v1.66.0 pythonpython 3.10 | 3.11 | 3.12 | 3.13 | 3.143.10 | 3.11 | 3.12 | 3.13 | 3.14 downloadsdownloads 63M/week63M/week
```
Documentation: https://googleapis.github.io/python-genai/
```
```
https://github.com/googleapis/python-genai
```
Google Gen AI Python SDK provides an interface for developers to integrate Google’s generative
models into their Python applications. It supports the Gemini Developer API and Vertex AI APIs.
Installation
pip install google-genai
With uv:
uv pip install google-genai
Imports
from google import genai
from google.genai import types
Create a client
```
Please run one of the following code blocks to create a client for different services (Gemini
```
```
Developer API or Vertex AI).
```
from google import genai
# Only run this block for Gemini Developer API
```
client = genai.Client(api_key='GEMINI_API_KEY')
```
from google import genai
# Only run this block for Vertex AI API
```
client = genai.Client(
```
```
vertexai=True, project='your-project-id', location='us-central1'
```
```
)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 1/152
```
```
(Optional) Using environment variables:
```
You can create a client by configuring the necessary environment variables. Configuration setup
instructions depends on whether you’re using the Gemini Developer API or the Gemini API in Vertex
AI.
Gemini Developer API: Set the GEMINI_API_KEY or GOOGLE_API_KEY. It will automatically be picked
up by the client. It’s recommended that you set only one of those variables, but if both are set,
GOOGLE_API_KEY takes precedence.
export GEMINI_API_KEY='your-api-key'
Gemini API on Vertex AI: Set GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT and
GOOGLE_CLOUD_LOCATION, as shown below:
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
from google import genai
```
client = genai.Client()
```
Close a client
Explicitly close the sync client to ensure that resources, such as the underlying HTTP connections,
are properly cleaned up and closed.
from google.genai import Client
```
client = Client()
```
```
response_1 = client.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Hello',
```
```
)
```
```
response_2 = client.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Ask a question',
```
```
)
```
# Close the sync client to release resources.
```
client.close()
```
To explicitly close the async client:
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 2/152
```
from google.genai import Client
```
aclient = Client(
```
```
vertexai=True, project='my-project-id', location='us-central1'
```
```
).aio
```
```
response_1 = await aclient.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Hello',
```
```
)
```
```
response_2 = await aclient.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Ask a question',
```
```
)
```
# Close the async client to release resources.
```
await aclient.aclose()
```
Client context managers
By using the sync client context manager, it will close the underlying sync client when exiting the
with block.
from google.genai import Client
```
with Client() as client:
```
```
response_1 = client.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Hello',
```
```
)
```
```
response_2 = client.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Ask a question',
```
```
)
```
By using the async client context manager, it will close the underlying async client when exiting the
with block.
from google.genai import Client
```
async with Client().aio as aclient:
```
```
response_1 = await aclient.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Hello',
```
```
)
```
```
response_2 = await aclient.models.generate_content(
```
```
model=MODEL_ID,
```
```
contents='Ask a question',
```
```
)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 3/152
```
API Selection
By default, the SDK uses the beta API endpoints provided by Google to support preview features in
the APIs. The stable API endpoints can be selected by setting the API version to v1.
To set the API version use http_options . For example, to set the API version to v1 for Vertex AI:
from google import genai
from google.genai import types
```
client = genai.Client(
```
```
vertexai=True,
```
```
project='your-project-id',
```
```
location='us-central1',
```
```
http_options=types.HttpOptions(api_version='v1')
```
```
)
```
To set the API version to v1alpha for the Gemini Developer API:
from google import genai
from google.genai import types
# Only run this block for Gemini Developer API
```
client = genai.Client(
```
```
api_key='GEMINI_API_KEY',
```
```
http_options=types.HttpOptions(api_version='v1alpha')
```
```
)
```
Faster async client option: Aiohttp
By default we use httpx for both sync and async client implementations. In order to have faster
performance, you may install google-genai[aiohttp]. In Gen AI SDK we configure trust_env=True to
```
match with the default behavior of httpx. Additional args of aiohttp.ClientSession.request() (see
```
```
_RequestOptions args) can be passed through the following way:
```
```
http_options = types.HttpOptions(
```
```
async_client_args={'cookies': ..., 'ssl': ...},
```
```
)
```
```
client=Client(..., http_options=http_options)
```
Proxy
Both httpx and aiohttp libraries use urllib.request.getproxies from environment variables. Before
```
client initialization, you may set proxy (and optional SSL_CERT_FILE) by setting the environment
```
```
variables:
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 4/152
```
export HTTPS_PROXY='http://username:password@proxy_uri:port'
export SSL_CERT_FILE='client.pem'
```
If you need socks5 proxy, httpx supports socks5 proxy if you pass it via args to httpx.Client(). You
```
may install httpx[socks] to use it. Then you can pass it through the following way:
```
http_options = types.HttpOptions(
```
```
client_args={'proxy': 'socks5://user:pass@host:port'},
```
```
async_client_args={'proxy': 'socks5://user:pass@host:port'},
```
```
)
```
```
client=Client(..., http_options=http_options)
```
Custom base url
```
In some cases you might need a custom base url (for example, API gateway proxy server) and
```
bypass some authentication checks for project, location, or API key. You may pass the custom base
url like this:
```
base_url = 'https://test-api-gateway-proxy.com'
```
```
client = Client(
```
```
vertexai=True, # Currently only vertexai=True is supported
```
```
http_options={
```
'base_url': base_url,
```
'headers': {'Authorization': 'Bearer test_token'},
```
```
},
```
```
)
```
Types
```
Parameter types can be specified as either dictionaries( TypedDict ) or Pydantic Models. Pydantic
```
model types are available in the types module.
Models
The client.models modules exposes model inferencing and model getters. See the ‘Create a client’
section above to initialize a client.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 5/152
```
Generate Content
```
with text content input (text output)
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash', contents='Why is the sky blue?'
```
```
)
```
```
print(response.text)
```
```
with text content input (image output)
```
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash-image',
```
```
contents='A cartoon infographic for flying sneakers',
```
```
config=types.GenerateContentConfig(
```
```
response_modalities=["IMAGE"],
```
```
image_config=types.ImageConfig(
```
```
aspect_ratio="9:16",
```
```
),
```
```
),
```
```
)
```
for part in response.parts:
if part.inline_data:
```
generated_image = part.as_image()
```
```
generated_image.show()
```
```
with uploaded file (Gemini Developer API only)
```
download the file in console.
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
python code.
```
file = client.files.upload(file='a11.txt')
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents=['Could you summarize this file?', file]
```
```
)
```
```
print(response.text)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 6/152
```
How to structure contents argument for generate_content
The SDK always converts the inputs to the contents argument into list[types.Content]. The following
shows some common ways to provide your inputs.
Provide a list[types.Content]
This is the canonical way to provide contents, SDK will not do any conversion.
Provide a types.Content instance
from google.genai import types
```
contents = types.Content(
```
```
role='user',
```
```
parts=[types.Part.from_text(text='Why is the sky blue?')]
```
```
)
```
SDK converts this to
[
```
types.Content(
```
```
role='user',
```
```
parts=[types.Part.from_text(text='Why is the sky blue?')]
```
```
)
```
]
Provide a string
```
contents='Why is the sky blue?'
```
The SDK will assume this is a text part, and it converts this into the following:
[
```
types.UserContent(
```
```
parts=[
```
```
types.Part.from_text(text='Why is the sky blue?')
```
]
```
)
```
]
Where a types.UserContent is a subclass of types.Content, it sets the role field to be user.
Provide a list of string
```
contents=['Why is the sky blue?', 'Why is the cloud white?']
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 7/152
```
The SDK assumes these are 2 text parts, it converts this into a single content, like the following:
[
```
types.UserContent(
```
```
parts=[
```
```
types.Part.from_text(text='Why is the sky blue?'),
```
```
types.Part.from_text(text='Why is the cloud white?'),
```
]
```
)
```
]
Where a types.UserContent is a subclass of types.Content, the role field in types.UserContent is fixed
to be user.
Provide a function call part
from google.genai import types
```
contents = types.Part.from_function_call(
```
```
name='get_weather_by_location',
```
```
args={'location': 'Boston'}
```
```
)
```
The SDK converts a function call part to a content with a model role:
[
```
types.ModelContent(
```
```
parts=[
```
```
types.Part.from_function_call(
```
```
name='get_weather_by_location',
```
```
args={'location': 'Boston'}
```
```
)
```
]
```
)
```
]
Where a types.ModelContent is a subclass of types.Content, the role field in types.ModelContent is
fixed to be model.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 8/152
```
Provide a list of function call parts
from google.genai import types
```
contents = [
```
```
types.Part.from_function_call(
```
```
name='get_weather_by_location',
```
```
args={'location': 'Boston'}
```
```
),
```
```
types.Part.from_function_call(
```
```
name='get_weather_by_location',
```
```
args={'location': 'New York'}
```
```
),
```
]
The SDK converts a list of function call parts to the a content with a model role:
[
```
types.ModelContent(
```
```
parts=[
```
```
types.Part.from_function_call(
```
```
name='get_weather_by_location',
```
```
args={'location': 'Boston'}
```
```
),
```
```
types.Part.from_function_call(
```
```
name='get_weather_by_location',
```
```
args={'location': 'New York'}
```
```
)
```
]
```
)
```
]
Where a types.ModelContent is a subclass of types.Content, the role field in types.ModelContent is
fixed to be model.
Provide a non function call part
from google.genai import types
```
contents = types.Part.from_uri(
```
```
file_uri: 'gs://generativeai-downloads/images/scones.jpg',
```
```
mime_type: 'image/jpeg',
```
```
)
```
The SDK converts all non function call parts into a content with a user role.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 9/152
```
[
```
types.UserContent(parts=[
```
```
types.Part.from_uri(
```
```
file_uri: 'gs://generativeai-downloads/images/scones.jpg',
```
```
mime_type: 'image/jpeg',
```
```
)
```
```
])
```
]
Provide a list of non function call parts
from google.genai import types
```
contents = [
```
```
types.Part.from_text('What is this image about?'),
```
```
types.Part.from_uri(
```
```
file_uri: 'gs://generativeai-downloads/images/scones.jpg',
```
```
mime_type: 'image/jpeg',
```
```
)
```
]
The SDK will convert the list of parts into a content with a user role
[
```
types.UserContent(
```
```
parts=[
```
```
types.Part.from_text('What is this image about?'),
```
```
types.Part.from_uri(
```
```
file_uri: 'gs://generativeai-downloads/images/scones.jpg',
```
```
mime_type: 'image/jpeg',
```
```
)
```
]
```
)
```
]
Mix types in contents
You can also provide a list of types.ContentUnion. The SDK leaves items of types.Content as is, it
groups consecutive non function call parts into a single types.UserContent, and it groups
consecutive function call parts into a single types.ModelContent.
If you put a list within a list, the inner list can only contain types.PartUnion items. The SDK will
convert the inner list into a single types.UserContent.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 10/152
```
System Instructions and Other Configs
The output of the model can be influenced by several optional settings available in
generate_content’s config parameter. For example, increasing max_output_tokens is essential for
longer model responses. To make a model more deterministic, lowering the temperature parameter
reduces randomness, with values near 0 minimizing variability. Capabilities and parameter defaults
for each model is shown in the Vertex AI docs and Gemini API docs respectively.
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.0-flash-001',
```
```
contents='high',
```
```
config=types.GenerateContentConfig(
```
```
system_instruction='I say high, you say low',
```
```
max_output_tokens=3,
```
```
temperature=0.3,
```
```
),
```
```
)
```
```
print(response.text)
```
Typed Config
All API methods support Pydantic types for parameters as well as dictionaries. You can get the type
from google.genai.types .
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.0-flash-001',
```
```
contents=types.Part.from_text(text='Why is the sky blue?'),
```
```
config=types.GenerateContentConfig(
```
```
temperature=0,
```
```
top_p=0.95,
```
```
top_k=20,
```
```
candidate_count=1,
```
```
seed=5,
```
```
max_output_tokens=100,
```
```
stop_sequences=['STOP!'],
```
```
presence_penalty=0.0,
```
```
frequency_penalty=0.0,
```
```
),
```
```
)
```
```
print(response.text)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 11/152
```
List Base Models
To retrieve tuned models, see: List Tuned Models
```
for model in client.models.list():
```
```
print(model)
```
```
pager = client.models.list(config={'page_size': 10})
```
```
print(pager.page_size)
```
```
print(pager[0])
```
```
pager.next_page()
```
```
print(pager[0])
```
```
List Base Models (Asynchronous)
```
```
async for job in await client.aio.models.list():
```
```
print(job)
```
```
async_pager = await client.aio.models.list(config={'page_size': 10})
```
```
print(async_pager.page_size)
```
```
print(async_pager[0])
```
```
await async_pager.next_page()
```
```
print(async_pager[0])
```
Safety Settings
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='Say something bad.',
```
```
config=types.GenerateContentConfig(
```
```
safety_settings=[
```
```
types.SafetySetting(
```
```
category='HARM_CATEGORY_HATE_SPEECH',
```
```
threshold='BLOCK_ONLY_HIGH',
```
```
)
```
]
```
),
```
```
)
```
```
print(response.text)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 12/152
```
Function Calling
Automatic Python function Support:
You can pass a Python function directly and it will be automatically called and responded by default.
from google.genai import types
```
def get_current_weather(location: str) -> str:
```
"""Returns the current weather.
```
Args:
```
```
location: The city and state, e.g. San Francisco, CA
```
"""
return 'sunny'
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='What is the weather like in Boston?',
```
```
config=types.GenerateContentConfig(
```
```
tools=[get_current_weather],
```
```
),
```
```
)
```
```
print(response.text)
```
Disabling automatic function calling
If you pass in a python function as a tool directly, and do not want automatic function calling, you
can disable automatic function calling as follows:
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='What is the weather like in Boston?',
```
```
config=types.GenerateContentConfig(
```
```
tools=[get_current_weather],
```
```
automatic_function_calling=types.AutomaticFunctionCallingConfig(
```
```
disable=True
```
```
),
```
```
),
```
```
)
```
With automatic function calling disabled, you will get a list of function call parts in the response:
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 13/152
```
```
function_calls: Optional[List[types.FunctionCall]] = response.function_calls
```
Manually declare and invoke a function for function calling
If you don’t want to use the automatic function support, you can manually declare the function and
invoke it.
The following example shows how to declare a function and pass it as a tool. Then you will receive
a function call part in the response.
from google.genai import types
```
function = types.FunctionDeclaration(
```
```
name='get_current_weather',
```
```
description='Get the current weather in a given location',
```
```
parameters_json_schema={
```
'type': 'object',
```
'properties': {
```
```
'location': {
```
'type': 'string',
'description': 'The city and state, e.g. San Francisco, CA',
```
}
```
```
},
```
'required': ['location'],
```
},
```
```
)
```
```
tool = types.Tool(function_declarations=[function])
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='What is the weather like in Boston?',
```
```
config=types.GenerateContentConfig(
```
```
tools=[tool],
```
```
),
```
```
)
```
```
print(response.function_calls[0])
```
After you receive the function call part from the model, you can invoke the function and get the
```
function response. And then you can pass the function response to the model. The following
```
example shows how to do it for a simple function invocation.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 14/152
```
from google.genai import types
```
user_prompt_content = types.Content(
```
```
role='user',
```
```
parts=[types.Part.from_text(text='What is the weather like in Boston?')],
```
```
)
```
```
function_call_part = response.function_calls[0]
```
```
function_call_content = response.candidates[0].content
```
```
try:
```
```
function_result = get_current_weather(
```
**function_call_part.function_call.args
```
)
```
```
function_response = {'result': function_result}
```
```
except (
```
Exception
```
) as e: # instead of raising the exception, you can let the model handle it
```
```
function_response = {'error': str(e)}
```
```
function_response_part = types.Part.from_function_response(
```
```
name=function_call_part.name,
```
```
response=function_response,
```
```
)
```
```
function_response_content = types.Content(
```
```
role='tool', parts=[function_response_part]
```
```
)
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents=[
```
user_prompt_content,
function_call_content,
function_response_content,
],
```
config=types.GenerateContentConfig(
```
```
tools=[tool],
```
```
),
```
```
)
```
```
print(response.text)
```
Function calling with ANY tools config mode
If you configure function calling mode to be ANY, then the model will always return function call
parts. If you also pass a python function as a tool, by default the SDK will perform automatic
```
function calling until the remote calls exceed the maximum remote call for automatic function
```
```
calling (default to 10 times).
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 15/152
```
If you’d like to disable automatic function calling in ANY mode:
from google.genai import types
```
def get_current_weather(location: str) -> str:
```
"""Returns the current weather.
```
Args:
```
```
location: The city and state, e.g. San Francisco, CA
```
"""
return "sunny"
```
response = client.models.generate_content(
```
```
model="gemini-2.5-flash",
```
```
contents="What is the weather like in Boston?",
```
```
config=types.GenerateContentConfig(
```
```
tools=[get_current_weather],
```
```
automatic_function_calling=types.AutomaticFunctionCallingConfig(
```
```
disable=True
```
```
),
```
```
tool_config=types.ToolConfig(
```
```
function_calling_config=types.FunctionCallingConfig(mode='ANY')
```
```
),
```
```
),
```
```
)
```
If you’d like to set x number of automatic function call turns, you can configure the maximum
remote calls to be x + 1 . Assuming you prefer 1 turn for automatic function calling:
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 16/152
```
from google.genai import types
```
def get_current_weather(location: str) -> str:
```
"""Returns the current weather.
```
Args:
```
```
location: The city and state, e.g. San Francisco, CA
```
"""
return "sunny"
```
response = client.models.generate_content(
```
```
model="gemini-2.5-flash",
```
```
contents="What is the weather like in Boston?",
```
```
config=types.GenerateContentConfig(
```
```
tools=[get_current_weather],
```
```
automatic_function_calling=types.AutomaticFunctionCallingConfig(
```
```
maximum_remote_calls=2
```
```
),
```
```
tool_config=types.ToolConfig(
```
```
function_calling_config=types.FunctionCallingConfig(mode='ANY')
```
```
),
```
```
),
```
```
)
```
```
Model Context Protocol (MCP) support (experimental)
```
Built-in MCP support is an experimental feature. You can pass a local MCP server as a tool directly.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 17/152
```
 
import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
```
client = genai.Client()
```
# Create server parameters for stdio connection
```
server_params = StdioServerParameters(
```
```
command="npx", # Executable
```
```
args=["-y", "@philschmid/weather-mcp"], # MCP Server
```
```
env=None, # Optional environment variables
```
```
)
```
```
async def run():
```
```
async with stdio_client(server_params) as (read, write):
```
```
async with ClientSession(read, write) as session:
```
# Prompt to get the weather for the current day in London.
```
prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%
```
# Initialize the connection between client and server
```
await session.initialize()
```
# Send request to the model with MCP function declarations
```
response = await client.aio.models.generate_content(
```
```
model="gemini-2.5-flash",
```
```
contents=prompt,
```
```
config=genai.types.GenerateContentConfig(
```
```
temperature=0,
```
```
tools=[session], # uses the session, will automatically call the too
```
```
),
```
```
)
```
```
print(response.text)
```
# Start the asyncio event loop and run the main function
```
asyncio.run(run())
```
JSON Response Schema
However you define your schema, don’t duplicate it in your input prompt, including by giving
examples of expected JSON output. If you do, the generated output might be lower in quality.
JSON Schema support
Schemas can be provided as standard JSON schema.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 18/152
```
```
user_profile = {
```
```
'properties': {
```
```
'age': {
```
'anyOf': [
```
{'maximum': 20, 'minimum': 0, 'type': 'integer'},
```
```
{'type': 'null'},
```
],
'title': 'Age',
```
},
```
```
'username': {
```
'description': "User's unique name",
'title': 'Username',
'type': 'string',
```
},
```
```
},
```
'required': ['username', 'age'],
'title': 'User Schema',
'type': 'object',
```
}
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='Give me a random user profile.',
```
```
config={
```
'response_mime_type': 'application/json',
'response_json_schema': user_profile
```
},
```
```
)
```
```
print(response.parsed)
```
Pydantic Model Schema support
Schemas can be provided as Pydantic Models.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 19/152
```
from pydantic import BaseModel
from google.genai import types
```
class CountryInfo(BaseModel):
```
```
name: str
```
```
population: int
```
```
capital: str
```
```
continent: str
```
```
gdp: int
```
```
official_language: str
```
```
total_area_sq_mi: int
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='Give me information for the United States.',
```
```
config=types.GenerateContentConfig(
```
```
response_mime_type='application/json',
```
```
response_schema=CountryInfo,
```
```
),
```
```
)
```
```
print(response.text)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 20/152
```
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='Give me information for the United States.',
```
```
config=types.GenerateContentConfig(
```
```
response_mime_type='application/json',
```
```
response_schema={
```
'required': [
'name',
'population',
'capital',
'continent',
'gdp',
'official_language',
'total_area_sq_mi',
],
```
'properties': {
```
```
'name': {'type': 'STRING'},
```
```
'population': {'type': 'INTEGER'},
```
```
'capital': {'type': 'STRING'},
```
```
'continent': {'type': 'STRING'},
```
```
'gdp': {'type': 'INTEGER'},
```
```
'official_language': {'type': 'STRING'},
```
```
'total_area_sq_mi': {'type': 'INTEGER'},
```
```
},
```
'type': 'OBJECT',
```
},
```
```
),
```
```
)
```
```
print(response.text)
```
Enum Response Schema
Text Response
You can set response_mime_type to 'text/x.enum' to return one of those enum values as the
response.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 21/152
```
from enum import Enum
```
class InstrumentEnum(Enum):
```
```
PERCUSSION = 'Percussion'
```
```
STRING = 'String'
```
```
WOODWIND = 'Woodwind'
```
```
BRASS = 'Brass'
```
```
KEYBOARD = 'Keyboard'
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='What instrument plays multiple notes at once?',
```
```
config={
```
'response_mime_type': 'text/x.enum',
'response_schema': InstrumentEnum,
```
},
```
```
)
```
```
print(response.text)
```
JSON Response
You can also set response_mime_type to 'application/json' , the response will be identical but in
quotes.
from enum import Enum
```
class InstrumentEnum(Enum):
```
```
PERCUSSION = 'Percussion'
```
```
STRING = 'String'
```
```
WOODWIND = 'Woodwind'
```
```
BRASS = 'Brass'
```
```
KEYBOARD = 'Keyboard'
```
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='What instrument plays multiple notes at once?',
```
```
config={
```
'response_mime_type': 'application/json',
'response_schema': InstrumentEnum,
```
},
```
```
)
```
```
print(response.text)
```
```
Generate Content (Synchronous Streaming)
```
Generate content in a streaming format so that the model outputs streams back to you, rather than
being returned as one chunk.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 22/152
```
Streaming for text content
```
for chunk in client.models.generate_content_stream(
```
```
model='gemini-2.5-flash', contents='Tell me a story in 300 words.'
```
```
):
```
```
print(chunk.text, end='')
```
Streaming for image content
If your image is stored in Google Cloud Storage, you can use the from_uri class method to create a
Part object.
from google.genai import types
```
for chunk in client.models.generate_content_stream(
```
```
model='gemini-2.5-flash',
```
```
contents=[
```
'What is this image about?',
```
types.Part.from_uri(
```
```
file_uri='gs://generativeai-downloads/images/scones.jpg',
```
```
mime_type='image/jpeg',
```
```
),
```
],
```
):
```
```
print(chunk.text, end='')
```
If your image is stored in your local file system, you can read it in as bytes data and use the
from_bytes class method to create a Part object.
from google.genai import types
```
YOUR_IMAGE_PATH = 'your_image_path'
```
```
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
```
```
with open(YOUR_IMAGE_PATH, 'rb') as f:
```
```
image_bytes = f.read()
```
```
for chunk in client.models.generate_content_stream(
```
```
model='gemini-2.5-flash',
```
```
contents=[
```
'What is this image about?',
```
types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
```
],
```
):
```
```
print(chunk.text, end='')Skip to content
```
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 23/152
```
```
Generate Content (Asynchronous Non Streaming)
```
client.aio exposes all the analogous async methods that are available on client . Note that it
applies to all the modules.
For example, client.aio.models.generate_content is the async version of
client.models.generate_content
```
response = await client.aio.models.generate_content(
```
```
model='gemini-2.5-flash', contents='Tell me a story in 300 words.'
```
```
)
```
```
print(response.text)
```
```
Generate Content (Asynchronous Streaming)
```
```
async for chunk in await client.aio.models.generate_content_stream(
```
```
model='gemini-2.5-flash', contents='Tell me a story in 300 words.'
```
```
):
```
```
print(chunk.text, end='')
```
Count Tokens and Compute Tokens
```
response = client.models.count_tokens(
```
```
model='gemini-2.5-flash',
```
```
contents='why is the sky blue?',
```
```
)
```
```
print(response)
```
Compute Tokens
Compute tokens is only supported in Vertex AI.
```
response = client.models.compute_tokens(
```
```
model='gemini-2.5-flash',
```
```
contents='why is the sky blue?',
```
```
)
```
```
print(response)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 24/152
```
```
Count Tokens (Asynchronous)
```
```
response = await client.aio.models.count_tokens(
```
```
model='gemini-2.5-flash',
```
```
contents='why is the sky blue?',
```
```
)
```
```
print(response)
```
Local Count Tokens
```
tokenizer = genai.LocalTokenizer(model_name='gemini-2.5-flash')
```
```
result = tokenizer.count_tokens("What is your name?")
```
Local Compute Tokens
```
tokenizer = genai.LocalTokenizer(model_name='gemini-2.5-flash')
```
```
result = tokenizer.compute_tokens("What is your name?")
```
Embed Content
```
response = client.models.embed_content(
```
```
model='gemini-embedding-001',
```
```
contents='why is the sky blue?',
```
```
)
```
```
print(response)
```
from google.genai import types
# multiple contents with config
```
response = client.models.embed_content(
```
```
model='gemini-embedding-001',
```
```
contents=['why is the sky blue?', 'What is your age?'],
```
```
config=types.EmbedContentConfig(output_dimensionality=10),
```
```
)
```
```
print(response)
```
Imagen
Generate Images
Support for generate images in Gemini Developer API is behind an allowlist
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 25/152
```
from google.genai import types
# Generate Image
```
response1 = client.models.generate_images(
```
```
model='imagen-3.0-generate-002',
```
```
prompt='An umbrella in the foreground, and a rainy night sky in the background',
```
```
config=types.GenerateImagesConfig(
```
```
number_of_images=1,
```
```
include_rai_reason=True,
```
```
output_mime_type='image/jpeg',
```
```
),
```
```
)
```
```
response1.generated_images[0].image.show()
```
Upscale Image
Upscale image is only supported in Vertex AI.
from google.genai import types
# Upscale the generated image from above
```
response2 = client.models.upscale_image(
```
```
model='imagen-3.0-generate-002',
```
```
image=response1.generated_images[0].image,
```
```
upscale_factor='x2',
```
```
config=types.UpscaleImageConfig(
```
```
include_rai_reason=True,
```
```
output_mime_type='image/jpeg',
```
```
),
```
```
)
```
```
response2.generated_images[0].image.show()
```
Edit Image
Edit image uses a separate model from generate and upscale.
Edit image is only supported in Vertex AI.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 26/152
```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage
```
raw_ref_image = RawReferenceImage(
```
```
reference_id=1,
```
```
reference_image=response1.generated_images[0].image,
```
```
)
```
# Model computes a mask of the background
```
mask_ref_image = MaskReferenceImage(
```
```
reference_id=2,
```
```
config=types.MaskReferenceConfig(
```
```
mask_mode='MASK_MODE_BACKGROUND',
```
```
mask_dilation=0,
```
```
),
```
```
)
```
```
response3 = client.models.edit_image(
```
```
model='imagen-3.0-capability-001',
```
```
prompt='Sunlight and clear sky',
```
```
reference_images=[raw_ref_image, mask_ref_image],
```
```
config=types.EditImageConfig(
```
```
edit_mode='EDIT_MODE_INPAINT_INSERTION',
```
```
number_of_images=1,
```
```
include_rai_reason=True,
```
```
output_mime_type='image/jpeg',
```
```
),
```
```
)
```
```
response3.generated_images[0].image.show()
```
Veo
Support for generating videos is considered public preview
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 27/152
```
```
Generate Videos (Text to Video)
```
from google.genai import types
# Create operation
```
operation = client.models.generate_videos(
```
```
model='veo-2.0-generate-001',
```
```
prompt='A neon hologram of a cat driving at top speed',
```
```
config=types.GenerateVideosConfig(
```
```
number_of_videos=1,
```
```
duration_seconds=5,
```
```
enhance_prompt=True,
```
```
),
```
```
)
```
# Poll operation
while not operation.done:
```
time.sleep(20)
```
```
operation = client.operations.get(operation)
```
```
video = operation.response.generated_videos[0].video
```
```
video.show()
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 28/152
```
```
Generate Videos (Image to Video)
```
from google.genai import types
```
# Read local image (uses mimetypes.guess_type to infer mime type)
```
```
image = types.Image.from_file("local/path/file.png")
```
# Create operation
```
operation = client.models.generate_videos(
```
```
model='veo-2.0-generate-001',
```
# Prompt is optional if image is provided
```
prompt='Night sky',
```
```
image=image,
```
```
config=types.GenerateVideosConfig(
```
```
number_of_videos=1,
```
```
duration_seconds=5,
```
```
enhance_prompt=True,
```
# Can also pass an Image into last_frame for frame interpolation
```
),
```
```
)
```
# Poll operation
while not operation.done:
```
time.sleep(20)
```
```
operation = client.operations.get(operation)
```
```
video = operation.response.generated_videos[0].video
```
```
video.show()
```
```
Generate Videos (Video to Video)
```
```
Currently, only Vertex AI supports Video to Video generation (Video extension).
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 29/152
```
from google.genai import types
```
# Read local video (uses mimetypes.guess_type to infer mime type)
```
```
video = types.Video.from_file("local/path/video.mp4")
```
# Create operation
```
operation = client.models.generate_videos(
```
```
model='veo-2.0-generate-001',
```
# Prompt is optional if Video is provided
```
prompt='Night sky',
```
# Input video must be in GCS
```
video=types.Video(
```
```
uri="gs://bucket-name/inputs/videos/cat_driving.mp4",
```
```
),
```
```
config=types.GenerateVideosConfig(
```
```
number_of_videos=1,
```
```
duration_seconds=5,
```
```
enhance_prompt=True,
```
```
),
```
```
)
```
# Poll operation
while not operation.done:
```
time.sleep(20)
```
```
operation = client.operations.get(operation)
```
```
video = operation.response.generated_videos[0].video
```
```
video.show()
```
Chats
Create a chat session to start a multi-turn conversations with the model. Then, use
chat.send_message function multiple times within the same chat session so that it can reflect on its
```
previous responses (i.e., engage in an ongoing conversation). See the ‘Create a client’ section above
```
to initialize a client.
```
Send Message (Synchronous Non-Streaming)
```
```
chat = client.chats.create(model='gemini-2.5-flash')
```
```
response = chat.send_message('tell me a story')
```
```
print(response.text)
```
```
response = chat.send_message('summarize the story you told me in 1 sentence')
```
```
print(response.text)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 30/152
```
```
Send Message (Synchronous Streaming)
```
```
chat = client.chats.create(model='gemini-2.5-flash')
```
```
for chunk in chat.send_message_stream('tell me a story'):
```
```
print(chunk.text)
```
```
Send Message (Asynchronous Non-Streaming)
```
```
chat = client.aio.chats.create(model='gemini-2.5-flash')
```
```
response = await chat.send_message('tell me a story')
```
```
print(response.text)
```
```
Send Message (Asynchronous Streaming)
```
```
chat = client.aio.chats.create(model='gemini-2.5-flash')
```
```
async for chunk in await chat.send_message_stream('tell me a story'):
```
```
print(chunk.text)
```
Files
Files are only supported in Gemini Developer API. See the ‘Create a client’ section above to initialize
a client.
gcloud storage cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
gcloud storage cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
Upload
```
file1 = client.files.upload(file='2312.11805v3.pdf')
```
```
file2 = client.files.upload(file='2403.05530.pdf')
```
```
print(file1)
```
```
print(file2)
```
Get
```
file1 = client.files.upload(file='2312.11805v3.pdf')
```
```
file_info = client.files.get(name=file1.name)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 31/152
```
Delete
```
file3 = client.files.upload(file='2312.11805v3.pdf')
```
```
client.files.delete(name=file3.name)
```
Caches
client.caches contains the control plane APIs for cached content.
See the ‘Create a client’ section above to initialize a client.
Create
from google.genai import types
if client.vertexai:
```
file_uris = [
```
'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
]
```
else:
```
```
file_uris = [file1.uri, file2.uri]
```
```
cached_content = client.caches.create(
```
```
model='gemini-2.5-flash',
```
```
config=types.CreateCachedContentConfig(
```
```
contents=[
```
```
types.Content(
```
```
role='user',
```
```
parts=[
```
```
types.Part.from_uri(
```
```
file_uri=file_uris[0], mime_type='application/pdf'
```
```
),
```
```
types.Part.from_uri(
```
```
file_uri=file_uris[1],
```
```
mime_type='application/pdf',
```
```
),
```
],
```
)
```
],
```
system_instruction='What is the sum of the two pdfs?',
```
```
display_name='test cache',
```
```
ttl='3600s',
```
```
),
```
```
)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 32/152
```
Get
```
cached_content = client.caches.get(name=cached_content.name)
```
Generate Content with Caches
from google.genai import types
```
response = client.models.generate_content(
```
```
model='gemini-2.5-flash',
```
```
contents='Summarize the pdfs',
```
```
config=types.GenerateContentConfig(
```
```
cached_content=cached_content.name,
```
```
),
```
```
)
```
```
print(response.text)
```
Tunings
client.tunings contains tuning job APIs and supports supervised fine tuning through tune . Only
supported in Vertex AI. See the ‘Create a client’ section above to initialize a client.
Tune
Vertex AI supports tuning from GCS source or from a Vertex AI Multimodal Dataset
from google.genai import types
```
model = 'gemini-2.5-flash'
```
```
training_dataset = types.TuningDataset(
```
# or gcs_uri=my_vertex_multimodal_dataset
```
gcs_uri='gs://your-gcs-bucket/your-tuning-data.jsonl',
```
```
)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 33/152
```
from google.genai import types
```
tuning_job = client.tunings.tune(
```
```
base_model=model,
```
```
training_dataset=training_dataset,
```
```
config=types.CreateTuningJobConfig(
```
```
epoch_count=1, tuned_model_display_name='test_dataset_examples model'
```
```
),
```
```
)
```
```
print(tuning_job)
```
Get Tuning Job
```
tuning_job = client.tunings.get(name=tuning_job.name)
```
```
print(tuning_job)
```
import time
```
completed_states = set(
```
[
'JOB_STATE_SUCCEEDED',
'JOB_STATE_FAILED',
'JOB_STATE_CANCELLED',
]
```
)
```
while tuning_job.state not in completed_states:
```
print(tuning_job.state)
```
```
tuning_job = client.tunings.get(name=tuning_job.name)
```
```
time.sleep(10)
```
Use Tuned Model
```
response = client.models.generate_content(
```
```
model=tuning_job.tuned_model.endpoint,
```
```
contents='why is the sky blue?',
```
```
)
```
```
print(response.text)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 34/152
```
Get Tuned Model
```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
```
```
print(tuned_model)
```
Update Tuned Model
from google.genai import types
```
tuned_model = client.models.update(
```
```
model=tuning_job.tuned_model.model,
```
```
config=types.UpdateModelConfig(
```
```
display_name='my tuned model', description='my tuned model description'
```
```
),
```
```
)
```
```
print(tuned_model)
```
List Tuned Models
To retrieve base models, see: List Base Models
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}):
```
```
print(model)
```
```
pager = client.models.list(config={'page_size': 10, 'query_base': False})
```
```
print(pager.page_size)
```
```
print(pager[0])
```
```
pager.next_page()
```
```
print(pager[0])
```
```
List Tuned Models (Asynchronous)
```
 
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': Fals
```
```
print(job)
```
 
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False})
```
```
print(async_pager.page_size)
```
```
print(async_pager[0])
```
```
await async_pager.next_page()
```
```
print(async_pager[0])
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 35/152
```
Update Tuned Model
from google.genai import types
```
model = pager[0]
```
```
model = client.models.update(
```
```
model=model.name,
```
```
config=types.UpdateModelConfig(
```
```
display_name='my tuned model', description='my tuned model description'
```
```
),
```
```
)
```
```
print(model)
```
List Tuning Jobs
```
for job in client.tunings.list(config={'page_size': 10}):
```
```
print(job)
```
```
pager = client.tunings.list(config={'page_size': 10})
```
```
print(pager.page_size)
```
```
print(pager[0])
```
```
pager.next_page()
```
```
print(pager[0])
```
```
List Tuning Jobs (Asynchronous):
```
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
```
```
print(job)
```
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
```
```
print(async_pager.page_size)
```
```
print(async_pager[0])
```
```
await async_pager.next_page()
```
```
print(async_pager[0])
```
Batch Prediction
Create a batch job. See the ‘Create a client’ section above to initialize a client.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 36/152
```
Create
Vertex AI client support using a BigQuery table or a GCS file as the source.
 
# Specify model and source file only, destination and job display name will be auto-popul
```
job = client.batches.create(
```
```
model='gemini-2.5-flash',
```
```
src='bq://my-project.my-dataset.my-table', # or "gs://path/to/input/data"
```
```
)
```
```
print(job)
```
Gemini Developer API
# Create a batch job with inlined requests
```
batch_job = client.batches.create(
```
```
model="gemini-2.5-flash",
```
```
src=[{
```
```
"contents": [{
```
```
"parts": [{
```
"text": "Hello!",
```
}],
```
"role": "user",
```
}],
```
```
"config": {"response_modalities": ["text"]},
```
```
}],
```
```
)
```
job
In order to create a batch job with file name. Need to upload a json file. For example
myrequests.json:
 
```
{"key":"request_1", "request": {"contents": [{"parts": [{"text":
```
```
"Explain how AI works in a few words"}]}], "generation_config": {"response_modalities":
```
```
{"key":"request_2", "request": {"contents": [{"parts": [{"text": "Explain how Crypto work
```
Then upload the file.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 37/152
```
# Upload a file to Gemini Developer API
```
file_name = client.files.upload(
```
```
file='myrequests.json',
```
```
config=types.UploadFileConfig(display_name='test-json'),
```
```
)
```
# Create a batch job with file name
```
batch_job = client.batches.create(
```
```
model="gemini-2.0-flash",
```
```
src="files/test-json",
```
```
)
```
# Get a job by name
```
job = client.batches.get(name=job.name)
```
job.state
```
completed_states = set(
```
[
'JOB_STATE_SUCCEEDED',
'JOB_STATE_FAILED',
'JOB_STATE_CANCELLED',
'JOB_STATE_PAUSED',
]
```
)
```
while job.state not in completed_states:
```
print(job.state)
```
```
job = client.batches.get(name=job.name)
```
```
time.sleep(30)
```
job
List
from google.genai import types
```
for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
```
```
print(job)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 38/152
```
List Batch Jobs with Pager
from google.genai import types
```
pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
```
```
print(pager.page_size)
```
```
print(pager[0])
```
```
pager.next_page()
```
```
print(pager[0])
```
```
List Batch Jobs (Asynchronous)
```
from google.genai import types
```
async for job in await client.aio.batches.list(
```
```
config=types.ListBatchJobsConfig(page_size=10)
```
```
):
```
```
print(job)
```
```
List Batch Jobs with Pager (Asynchronous)
```
from google.genai import types
```
async_pager = await client.aio.batches.list(
```
```
config=types.ListBatchJobsConfig(page_size=10)
```
```
)
```
```
print(async_pager.page_size)
```
```
print(async_pager[0])
```
```
await async_pager.next_page()
```
```
print(async_pager[0])
```
Delete
# Delete the job resource
```
delete_job = client.batches.delete(name=job.name)
```
delete_job
Error Handling
To handle errors raised by the model service, the SDK provides this APIError class.
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 39/152
```
from google.genai import errors
```
try:
```
```
client.models.generate_content(
```
```
model="invalid-model-name",
```
```
contents="What is your name?",
```
```
)
```
except errors.APIError as e:
```
print(e.code) # 404
```
```
print(e.message)
```
Extra Request Body
The extra_body field in HttpOptions accepts a dictionary of additional JSON properties to include
in the request body. This can be used to access new or experimental backend features that are not
yet formally supported in the SDK. The structure of the dictionary must match the backend API’s
request structure.
VertexAI backend API docs: https://cloud.google.com/vertex-ai/docs/reference/rest
GeminiAPI backend API docs: https://ai.google.dev/api/rest
 
```
response = client.models.generate_content(
```
```
model="gemini-2.5-pro",
```
```
contents="What is the weather in Boston? and how about Sunnyvale?",
```
```
config=types.GenerateContentConfig(
```
```
tools=[get_current_weather],
```
```
http_options=types.HttpOptions(extra_body={'tool_config': {'function_calling_conf
```
```
),
```
```
)
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 40/152
```
Reference
Submodules
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 41/152
```
genai.client module
AsyncClient
```
AsyncClient.aclose()
```
AsyncClient.auth_tokens
AsyncClient.batches
AsyncClient.caches
AsyncClient.chats
AsyncClient.file_search_stores
AsyncClient.files
AsyncClient.interactions
AsyncClient.live
AsyncClient.models
AsyncClient.operations
AsyncClient.tunings
AsyncGeminiNextGenAPIClientAdapter
```
AsyncGeminiNextGenAPIClientAdapter.async_get_auth_headers()
```
```
AsyncGeminiNextGenAPIClientAdapter.get_location()
```
```
AsyncGeminiNextGenAPIClientAdapter.get_project()
```
```
AsyncGeminiNextGenAPIClientAdapter.is_vertex_ai()
```
Client
Client.api_key
Client.vertexai
Client.credentials
Client.project
Client.location
Client.debug_config
Client.http_options
Client.aio
Client.auth_tokens
Client.batches
Client.caches
Client.chats
```
Client.close()
```
Client.file_search_stores
Client.files
Client.interactions
Client.models
Client.operations
Client.tunings
Client.vertexai
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 42/152
```
DebugConfig
DebugConfig.client_mode
DebugConfig.replay_id
DebugConfig.replays_directory
GeminiNextGenAPIClientAdapter
```
GeminiNextGenAPIClientAdapter.get_auth_headers()
```
```
GeminiNextGenAPIClientAdapter.get_location()
```
```
GeminiNextGenAPIClientAdapter.get_project()
```
```
GeminiNextGenAPIClientAdapter.is_vertex_ai()
```
genai.batches module
AsyncBatches
```
AsyncBatches.cancel()
```
```
AsyncBatches.create()
```
```
AsyncBatches.create_embeddings()
```
```
AsyncBatches.delete()
```
```
AsyncBatches.get()
```
```
AsyncBatches.list()
```
Batches
```
Batches.cancel()
```
```
Batches.create()
```
```
Batches.create_embeddings()
```
```
Batches.delete()
```
```
Batches.get()
```
```
Batches.list()
```
genai.caches module
AsyncCaches
```
AsyncCaches.create()
```
```
AsyncCaches.delete()
```
```
AsyncCaches.get()
```
```
AsyncCaches.list()
```
```
AsyncCaches.update()
```
Caches
```
Caches.create()
```
```
Caches.delete()
```
```
Caches.get()
```
```
Caches.list()
```
```
Caches.update()
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 43/152
```
genai.chats module
AsyncChat
```
AsyncChat.send_message()
```
```
AsyncChat.send_message_stream()
```
AsyncChats
```
AsyncChats.create()
```
Chat
```
Chat.send_message()
```
```
Chat.send_message_stream()
```
Chats
```
Chats.create()
```
genai.file_search_stores module
AsyncFileSearchStores
```
AsyncFileSearchStores.create()
```
```
AsyncFileSearchStores.delete()
```
AsyncFileSearchStores.documents
```
AsyncFileSearchStores.get()
```
```
AsyncFileSearchStores.import_file()
```
```
AsyncFileSearchStores.list()
```
```
AsyncFileSearchStores.upload_to_file_search_store()
```
FileSearchStores
```
FileSearchStores.create()
```
```
FileSearchStores.delete()
```
FileSearchStores.documents
```
FileSearchStores.get()
```
```
FileSearchStores.import_file()
```
```
FileSearchStores.list()
```
```
FileSearchStores.upload_to_file_search_store()
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 44/152
```
genai.files module
AsyncFiles
```
AsyncFiles.delete()
```
```
AsyncFiles.download()
```
```
AsyncFiles.get()
```
```
AsyncFiles.list()
```
```
AsyncFiles.register_files()
```
```
AsyncFiles.upload()
```
Files
```
Files.delete()
```
```
Files.download()
```
```
Files.get()
```
```
Files.list()
```
```
Files.register_files()
```
```
Files.upload()
```
genai.interactions module
AsyncInteractionsResource
```
AsyncInteractionsResource.cancel()
```
```
AsyncInteractionsResource.create()
```
```
AsyncInteractionsResource.delete()
```
```
AsyncInteractionsResource.get()
```
AsyncInteractionsResource.with_raw_response
AsyncInteractionsResource.with_streaming_response
InteractionsResource
```
InteractionsResource.cancel()
```
```
InteractionsResource.create()
```
```
InteractionsResource.delete()
```
```
InteractionsResource.get()
```
InteractionsResource.with_raw_response
InteractionsResource.with_streaming_response
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 45/152
```
genai.live module
AsyncLive
```
AsyncLive.connect()
```
AsyncLive.music
AsyncSession
```
AsyncSession.close()
```
```
AsyncSession.receive()
```
```
AsyncSession.send()
```
```
AsyncSession.send_client_content()
```
```
AsyncSession.send_realtime_input()
```
```
AsyncSession.send_tool_response()
```
```
AsyncSession.start_stream()
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 46/152
```
genai.models module
AsyncModels
```
AsyncModels.compute_tokens()
```
```
AsyncModels.count_tokens()
```
```
AsyncModels.delete()
```
```
AsyncModels.edit_image()
```
```
AsyncModels.embed_content()
```
```
AsyncModels.generate_content()
```
```
AsyncModels.generate_content_stream()
```
```
AsyncModels.generate_images()
```
```
AsyncModels.generate_videos()
```
```
AsyncModels.get()
```
```
AsyncModels.list()
```
```
AsyncModels.recontext_image()
```
```
AsyncModels.segment_image()
```
```
AsyncModels.update()
```
```
AsyncModels.upscale_image()
```
Models
```
Models.compute_tokens()
```
```
Models.count_tokens()
```
```
Models.delete()
```
```
Models.edit_image()
```
```
Models.embed_content()
```
```
Models.generate_content()
```
```
Models.generate_content_stream()
```
```
Models.generate_images()
```
```
Models.generate_videos()
```
```
Models.get()
```
```
Models.list()
```
```
Models.recontext_image()
```
```
Models.segment_image()
```
```
Models.update()
```
```
Models.upscale_image()
```
genai.tokens module
AsyncTokens
```
AsyncTokens.create()
```
Tokens
```
Tokens.create()Skip to content
```
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 47/152
```
genai.tunings module
AsyncTunings
```
AsyncTunings.cancel()
```
```
AsyncTunings.get()
```
```
AsyncTunings.list()
```
```
AsyncTunings.tune()
```
Tunings
```
Tunings.cancel()
```
```
Tunings.get()
```
```
Tunings.list()
```
```
Tunings.tune()
```
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 48/152
```
genai.types module
ActivityEnd
ActivityEndDict
ActivityHandling
ActivityHandling.ACTIVITY_HANDLING_UNSPECIFIED
ActivityHandling.NO_INTERRUPTION
ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
ActivityStart
ActivityStartDict
AdapterSize
AdapterSize.ADAPTER_SIZE_EIGHT
AdapterSize.ADAPTER_SIZE_FOUR
AdapterSize.ADAPTER_SIZE_ONE
AdapterSize.ADAPTER_SIZE_SIXTEEN
AdapterSize.ADAPTER_SIZE_THIRTY_TWO
AdapterSize.ADAPTER_SIZE_TWO
AdapterSize.ADAPTER_SIZE_UNSPECIFIED
ApiAuth
ApiAuth.api_key_config
ApiAuthApiKeyConfig
ApiAuthApiKeyConfig.api_key_secret_version
ApiAuthApiKeyConfig.api_key_string
ApiAuthApiKeyConfigDict
ApiAuthApiKeyConfigDict.api_key_secret_version
ApiAuthApiKeyConfigDict.api_key_string
ApiAuthDict
ApiAuthDict.api_key_config
ApiKeyConfig
ApiKeyConfig.api_key_secret
ApiKeyConfig.api_key_string
ApiKeyConfig.http_element_location
ApiKeyConfig.name
ApiKeyConfigDict
ApiKeyConfigDict.api_key_secret
ApiKeyConfigDict.api_key_string
ApiKeyConfigDict.http_element_location
ApiKeyConfigDict.name
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 49/152
```
ApiSpec
ApiSpec.API_SPEC_UNSPECIFIED
ApiSpec.ELASTIC_SEARCH
ApiSpec.SIMPLE_SEARCH
AudioChunk
AudioChunk.data
AudioChunk.mime_type
AudioChunk.source_metadata
AudioChunkDict
AudioChunkDict.data
AudioChunkDict.mime_type
AudioChunkDict.source_metadata
AudioTranscriptionConfig
AudioTranscriptionConfigDict
AuthConfig
AuthConfig.api_key_config
AuthConfig.auth_type
AuthConfig.google_service_account_config
AuthConfig.http_basic_auth_config
AuthConfig.oauth_config
AuthConfig.oidc_config
AuthConfigDict
AuthConfigDict.api_key_config
AuthConfigDict.auth_type
AuthConfigDict.google_service_account_config
AuthConfigDict.http_basic_auth_config
AuthConfigDict.oauth_config
AuthConfigDict.oidc_config
AuthConfigGoogleServiceAccountConfig
AuthConfigGoogleServiceAccountConfig.service_account
AuthConfigGoogleServiceAccountConfigDict
AuthConfigGoogleServiceAccountConfigDict.service_account
AuthConfigHttpBasicAuthConfig
AuthConfigHttpBasicAuthConfig.credential_secret
AuthConfigHttpBasicAuthConfigDict
AuthConfigHttpBasicAuthConfigDict.credential_secret
AuthConfigOauthConfig
AuthConfigOauthConfig.access_token
AuthConfigOauthConfig.service_account
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 50/152
```
AuthConfigOauthConfigDict
AuthConfigOauthConfigDict.access_token
AuthConfigOauthConfigDict.service_account
AuthConfigOidcConfig
AuthConfigOidcConfig.id_token
AuthConfigOidcConfig.service_account
AuthConfigOidcConfigDict
AuthConfigOidcConfigDict.id_token
AuthConfigOidcConfigDict.service_account
AuthToken
AuthToken.name
AuthTokenDict
AuthTokenDict.name
AuthType
AuthType.API_KEY_AUTH
AuthType.AUTH_TYPE_UNSPECIFIED
AuthType.GOOGLE_SERVICE_ACCOUNT_AUTH
AuthType.HTTP_BASIC_AUTH
AuthType.NO_AUTH
AuthType.OAUTH
AuthType.OIDC_AUTH
AutomaticActivityDetection
AutomaticActivityDetection.disabled
AutomaticActivityDetection.end_of_speech_sensitivity
AutomaticActivityDetection.prefix_padding_ms
AutomaticActivityDetection.silence_duration_ms
AutomaticActivityDetection.start_of_speech_sensitivity
AutomaticActivityDetectionDict
AutomaticActivityDetectionDict.disabled
AutomaticActivityDetectionDict.end_of_speech_sensitivity
AutomaticActivityDetectionDict.prefix_padding_ms
AutomaticActivityDetectionDict.silence_duration_ms
AutomaticActivityDetectionDict.start_of_speech_sensitivity
AutomaticFunctionCallingConfig
AutomaticFunctionCallingConfig.disable
AutomaticFunctionCallingConfig.ignore_call_history
AutomaticFunctionCallingConfig.maximum_remote_calls
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 51/152
```
AutomaticFunctionCallingConfigDict
AutomaticFunctionCallingConfigDict.disable
AutomaticFunctionCallingConfigDict.ignore_call_history
AutomaticFunctionCallingConfigDict.maximum_remote_calls
AutoraterConfig
AutoraterConfig.autorater_model
AutoraterConfig.flip_enabled
AutoraterConfig.generation_config
AutoraterConfig.sampling_count
AutoraterConfigDict
AutoraterConfigDict.autorater_model
AutoraterConfigDict.flip_enabled
AutoraterConfigDict.generation_config
AutoraterConfigDict.sampling_count
BatchJob
BatchJob.completion_stats
BatchJob.create_time
BatchJob.dest
BatchJob.display_name
BatchJob.end_time
BatchJob.error
BatchJob.model
BatchJob.name
BatchJob.src
BatchJob.start_time
BatchJob.state
BatchJob.update_time
BatchJob.done
BatchJobDestination
BatchJobDestination.bigquery_uri
BatchJobDestination.file_name
BatchJobDestination.format
BatchJobDestination.gcs_uri
BatchJobDestination.inlined_embed_content_responses
BatchJobDestination.inlined_responses
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 52/152
```
BatchJobDestinationDict
BatchJobDestinationDict.bigquery_uri
BatchJobDestinationDict.file_name
BatchJobDestinationDict.format
BatchJobDestinationDict.gcs_uri
BatchJobDestinationDict.inlined_embed_content_responses
BatchJobDestinationDict.inlined_responses
BatchJobDict
BatchJobDict.completion_stats
BatchJobDict.create_time
BatchJobDict.dest
BatchJobDict.display_name
BatchJobDict.end_time
BatchJobDict.error
BatchJobDict.model
BatchJobDict.name
BatchJobDict.src
BatchJobDict.start_time
BatchJobDict.state
BatchJobDict.update_time
BatchJobSource
BatchJobSource.bigquery_uri
BatchJobSource.file_name
BatchJobSource.format
BatchJobSource.gcs_uri
BatchJobSource.inlined_requests
BatchJobSourceDict
BatchJobSourceDict.bigquery_uri
BatchJobSourceDict.file_name
BatchJobSourceDict.format
BatchJobSourceDict.gcs_uri
BatchJobSourceDict.inlined_requests
Behavior
Behavior.BLOCKING
Behavior.NON_BLOCKING
Behavior.UNSPECIFIED
BleuSpec
BleuSpec.use_effective_order
BleuSpecDict
BleuSpecDict.use_effective_order
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 53/152
```
Blob
Blob.data
Blob.display_name
Blob.mime_type
```
Blob.as_image()
```
BlobDict
BlobDict.data
BlobDict.display_name
BlobDict.mime_type
BlockedReason
BlockedReason.BLOCKED_REASON_UNSPECIFIED
BlockedReason.BLOCKLIST
BlockedReason.IMAGE_SAFETY
BlockedReason.JAILBREAK
BlockedReason.MODEL_ARMOR
BlockedReason.OTHER
BlockedReason.PROHIBITED_CONTENT
BlockedReason.SAFETY
CachedContent
CachedContent.create_time
CachedContent.display_name
CachedContent.expire_time
CachedContent.model
CachedContent.name
CachedContent.update_time
CachedContent.usage_metadata
CachedContentDict
CachedContentDict.create_time
CachedContentDict.display_name
CachedContentDict.expire_time
CachedContentDict.model
CachedContentDict.name
CachedContentDict.update_time
CachedContentDict.usage_metadata
CachedContentUsageMetadata
CachedContentUsageMetadata.audio_duration_seconds
CachedContentUsageMetadata.image_count
CachedContentUsageMetadata.text_count
CachedContentUsageMetadata.total_token_count
CachedContentUsageMetadata.video_duration_seconds
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 54/152
```
CachedContentUsageMetadataDict
CachedContentUsageMetadataDict.audio_duration_seconds
CachedContentUsageMetadataDict.image_count
CachedContentUsageMetadataDict.text_count
CachedContentUsageMetadataDict.total_token_count
CachedContentUsageMetadataDict.video_duration_seconds
CancelBatchJobConfig
CancelBatchJobConfig.http_options
CancelBatchJobConfigDict
CancelBatchJobConfigDict.http_options
CancelTuningJobConfig
CancelTuningJobConfig.http_options
CancelTuningJobConfigDict
CancelTuningJobConfigDict.http_options
CancelTuningJobResponse
CancelTuningJobResponse.sdk_http_response
CancelTuningJobResponseDict
CancelTuningJobResponseDict.sdk_http_response
Candidate
Candidate.avg_logprobs
Candidate.citation_metadata
Candidate.content
Candidate.finish_message
Candidate.finish_reason
Candidate.grounding_metadata
Candidate.index
Candidate.logprobs_result
Candidate.safety_ratings
Candidate.token_count
Candidate.url_context_metadata
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 55/152
```
CandidateDict
CandidateDict.avg_logprobs
CandidateDict.citation_metadata
CandidateDict.content
CandidateDict.finish_message
CandidateDict.finish_reason
CandidateDict.grounding_metadata
CandidateDict.index
CandidateDict.logprobs_result
CandidateDict.safety_ratings
CandidateDict.token_count
CandidateDict.url_context_metadata
Checkpoint
Checkpoint.checkpoint_id
Checkpoint.epoch
Checkpoint.step
CheckpointDict
CheckpointDict.checkpoint_id
CheckpointDict.epoch
CheckpointDict.step
ChunkingConfig
ChunkingConfig.white_space_config
ChunkingConfigDict
ChunkingConfigDict.white_space_config
Citation
Citation.end_index
Citation.license
Citation.publication_date
Citation.start_index
Citation.title
Citation.uri
CitationDict
CitationDict.end_index
CitationDict.license
CitationDict.publication_date
CitationDict.start_index
CitationDict.title
CitationDict.uri
CitationMetadata
CitationMetadata.citations
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 56/152
```
CitationMetadataDict
CitationMetadataDict.citations
CodeExecutionResult
CodeExecutionResult.outcome
CodeExecutionResult.output
CodeExecutionResultDict
CodeExecutionResultDict.outcome
CodeExecutionResultDict.output
CompletionStats
CompletionStats.failed_count
CompletionStats.incomplete_count
CompletionStats.successful_count
CompletionStats.successful_forecast_point_count
CompletionStatsDict
CompletionStatsDict.failed_count
CompletionStatsDict.incomplete_count
CompletionStatsDict.successful_count
CompletionStatsDict.successful_forecast_point_count
ComputationBasedMetricSpec
ComputationBasedMetricSpec.parameters
ComputationBasedMetricSpec.type
ComputationBasedMetricSpecDict
ComputationBasedMetricSpecDict.parameters
ComputationBasedMetricSpecDict.type
ComputationBasedMetricType
ComputationBasedMetricType.BLEU
ComputationBasedMetricType.COMPUTATION_BASED_METRIC_TYPE_UNSPECIFIED
ComputationBasedMetricType.EXACT_MATCH
ComputationBasedMetricType.ROUGE
ComputeTokensConfig
ComputeTokensConfig.http_options
ComputeTokensConfigDict
ComputeTokensConfigDict.http_options
ComputeTokensResponse
ComputeTokensResponse.sdk_http_response
ComputeTokensResponse.tokens_info
ComputeTokensResponseDict
ComputeTokensResponseDict.sdk_http_response
ComputeTokensResponseDict.tokens_info
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 57/152
```
ComputeTokensResult
ComputeTokensResult.tokens_info
ComputeTokensResultDict
ComputeTokensResultDict.tokens_info
ComputerUse
ComputerUse.environment
ComputerUse.excluded_predefined_functions
ComputerUseDict
ComputerUseDict.environment
ComputerUseDict.excluded_predefined_functions
Content
Content.parts
Content.role
ContentDict
ContentDict.parts
ContentDict.role
ContentEmbedding
ContentEmbedding.statistics
ContentEmbedding.values
ContentEmbeddingDict
ContentEmbeddingDict.statistics
ContentEmbeddingStatistics
ContentEmbeddingStatistics.token_count
ContentEmbeddingStatistics.truncated
ContentEmbeddingStatisticsDict
ContentEmbeddingStatisticsDict.token_count
ContentEmbeddingStatisticsDict.truncated
ContentReferenceImage
ContentReferenceImage.reference_id
ContentReferenceImage.reference_image
ContentReferenceImage.reference_type
ContentReferenceImageDict
ContentReferenceImageDict.reference_id
ContentReferenceImageDict.reference_image
ContentReferenceImageDict.reference_type
ContextWindowCompressionConfig
ContextWindowCompressionConfig.sliding_window
ContextWindowCompressionConfig.trigger_tokensSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 58/152
```
ContextWindowCompressionConfigDict
ContextWindowCompressionConfigDict.sliding_window
ContextWindowCompressionConfigDict.trigger_tokens
ControlReferenceConfig
ControlReferenceConfig.control_type
ControlReferenceConfig.enable_control_image_computation
ControlReferenceConfigDict
ControlReferenceConfigDict.control_type
ControlReferenceConfigDict.enable_control_image_computation
ControlReferenceImage
ControlReferenceImage.config
ControlReferenceImage.control_image_config
ControlReferenceImage.reference_id
ControlReferenceImage.reference_image
ControlReferenceImage.reference_type
ControlReferenceImageDict
ControlReferenceImageDict.config
ControlReferenceImageDict.reference_id
ControlReferenceImageDict.reference_image
ControlReferenceImageDict.reference_type
ControlReferenceType
ControlReferenceType.CONTROL_TYPE_CANNY
ControlReferenceType.CONTROL_TYPE_DEFAULT
ControlReferenceType.CONTROL_TYPE_FACE_MESH
ControlReferenceType.CONTROL_TYPE_SCRIBBLE
CountTokensConfig
CountTokensConfig.generation_config
CountTokensConfig.http_options
CountTokensConfig.system_instruction
CountTokensConfig.tools
CountTokensConfigDict
CountTokensConfigDict.generation_config
CountTokensConfigDict.http_options
CountTokensConfigDict.system_instruction
CountTokensConfigDict.tools
CountTokensResponse
CountTokensResponse.cached_content_token_count
CountTokensResponse.sdk_http_response
CountTokensResponse.total_tokens
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 59/152
```
CountTokensResponseDict
CountTokensResponseDict.cached_content_token_count
CountTokensResponseDict.sdk_http_response
CountTokensResponseDict.total_tokens
CountTokensResult
CountTokensResult.total_tokens
CountTokensResultDict
CountTokensResultDict.total_tokens
CreateAuthTokenConfig
CreateAuthTokenConfig.expire_time
CreateAuthTokenConfig.http_options
CreateAuthTokenConfig.live_connect_constraints
CreateAuthTokenConfig.lock_additional_fields
CreateAuthTokenConfig.new_session_expire_time
CreateAuthTokenConfig.uses
CreateAuthTokenConfigDict
CreateAuthTokenConfigDict.expire_time
CreateAuthTokenConfigDict.http_options
CreateAuthTokenConfigDict.live_connect_constraints
CreateAuthTokenConfigDict.lock_additional_fields
CreateAuthTokenConfigDict.new_session_expire_time
CreateAuthTokenConfigDict.uses
CreateAuthTokenParameters
CreateAuthTokenParameters.config
CreateAuthTokenParametersDict
CreateAuthTokenParametersDict.config
CreateBatchJobConfig
CreateBatchJobConfig.dest
CreateBatchJobConfig.display_name
CreateBatchJobConfig.http_options
CreateBatchJobConfigDict
CreateBatchJobConfigDict.dest
CreateBatchJobConfigDict.display_name
CreateBatchJobConfigDict.http_options
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 60/152
```
CreateCachedContentConfig
CreateCachedContentConfig.contents
CreateCachedContentConfig.display_name
CreateCachedContentConfig.expire_time
CreateCachedContentConfig.http_options
CreateCachedContentConfig.kms_key_name
CreateCachedContentConfig.system_instruction
CreateCachedContentConfig.tool_config
CreateCachedContentConfig.tools
CreateCachedContentConfig.ttl
CreateCachedContentConfigDict
CreateCachedContentConfigDict.contents
CreateCachedContentConfigDict.display_name
CreateCachedContentConfigDict.expire_time
CreateCachedContentConfigDict.http_options
CreateCachedContentConfigDict.kms_key_name
CreateCachedContentConfigDict.system_instruction
CreateCachedContentConfigDict.tool_config
CreateCachedContentConfigDict.tools
CreateCachedContentConfigDict.ttl
CreateEmbeddingsBatchJobConfig
CreateEmbeddingsBatchJobConfig.display_name
CreateEmbeddingsBatchJobConfig.http_options
CreateEmbeddingsBatchJobConfigDict
CreateEmbeddingsBatchJobConfigDict.display_name
CreateEmbeddingsBatchJobConfigDict.http_options
CreateFileConfig
CreateFileConfig.http_options
CreateFileConfig.should_return_http_response
CreateFileConfigDict
CreateFileConfigDict.http_options
CreateFileConfigDict.should_return_http_response
CreateFileResponse
CreateFileResponse.sdk_http_response
CreateFileResponseDict
CreateFileResponseDict.sdk_http_response
CreateFileSearchStoreConfig
CreateFileSearchStoreConfig.display_name
CreateFileSearchStoreConfig.http_options
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 61/152
```
CreateFileSearchStoreConfigDict
CreateFileSearchStoreConfigDict.display_name
CreateFileSearchStoreConfigDict.http_options
CreateTuningJobConfig
CreateTuningJobConfig.adapter_size
CreateTuningJobConfig.base_teacher_model
CreateTuningJobConfig.batch_size
CreateTuningJobConfig.beta
CreateTuningJobConfig.custom_base_model
CreateTuningJobConfig.description
CreateTuningJobConfig.encryption_spec
CreateTuningJobConfig.epoch_count
CreateTuningJobConfig.evaluation_config
CreateTuningJobConfig.export_last_checkpoint_only
CreateTuningJobConfig.http_options
CreateTuningJobConfig.labels
CreateTuningJobConfig.learning_rate
CreateTuningJobConfig.learning_rate_multiplier
CreateTuningJobConfig.method
CreateTuningJobConfig.output_uri
CreateTuningJobConfig.pre_tuned_model_checkpoint_id
CreateTuningJobConfig.sft_loss_weight_multiplier
CreateTuningJobConfig.tuned_model_display_name
CreateTuningJobConfig.tuned_teacher_model_source
CreateTuningJobConfig.tuning_mode
CreateTuningJobConfig.validation_dataset
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 62/152
```
CreateTuningJobConfigDict
CreateTuningJobConfigDict.adapter_size
CreateTuningJobConfigDict.base_teacher_model
CreateTuningJobConfigDict.batch_size
CreateTuningJobConfigDict.beta
CreateTuningJobConfigDict.custom_base_model
CreateTuningJobConfigDict.description
CreateTuningJobConfigDict.encryption_spec
CreateTuningJobConfigDict.epoch_count
CreateTuningJobConfigDict.evaluation_config
CreateTuningJobConfigDict.export_last_checkpoint_only
CreateTuningJobConfigDict.http_options
CreateTuningJobConfigDict.labels
CreateTuningJobConfigDict.learning_rate
CreateTuningJobConfigDict.learning_rate_multiplier
CreateTuningJobConfigDict.method
CreateTuningJobConfigDict.output_uri
CreateTuningJobConfigDict.pre_tuned_model_checkpoint_id
CreateTuningJobConfigDict.sft_loss_weight_multiplier
CreateTuningJobConfigDict.tuned_model_display_name
CreateTuningJobConfigDict.tuned_teacher_model_source
CreateTuningJobConfigDict.tuning_mode
CreateTuningJobConfigDict.validation_dataset
CreateTuningJobParameters
CreateTuningJobParameters.base_model
CreateTuningJobParameters.config
CreateTuningJobParameters.training_dataset
CreateTuningJobParametersDict
CreateTuningJobParametersDict.base_model
CreateTuningJobParametersDict.config
CreateTuningJobParametersDict.training_dataset
CustomCodeExecutionSpec
CustomCodeExecutionSpec.evaluation_function
CustomCodeExecutionSpecDict
CustomCodeExecutionSpecDict.evaluation_function
CustomMetadata
CustomMetadata.key
CustomMetadata.numeric_value
CustomMetadata.string_list_value
CustomMetadata.string_value
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 63/152
```
CustomMetadataDict
CustomMetadataDict.key
CustomMetadataDict.numeric_value
CustomMetadataDict.string_list_value
CustomMetadataDict.string_value
CustomOutputFormatConfig
CustomOutputFormatConfig.return_raw_output
CustomOutputFormatConfigDict
CustomOutputFormatConfigDict.return_raw_output
DatasetDistribution
DatasetDistribution.buckets
DatasetDistribution.max
DatasetDistribution.mean
DatasetDistribution.median
DatasetDistribution.min
DatasetDistribution.p5
DatasetDistribution.p95
DatasetDistribution.sum
DatasetDistributionDict
DatasetDistributionDict.buckets
DatasetDistributionDict.max
DatasetDistributionDict.mean
DatasetDistributionDict.median
DatasetDistributionDict.min
DatasetDistributionDict.p5
DatasetDistributionDict.p95
DatasetDistributionDict.sum
DatasetDistributionDistributionBucket
DatasetDistributionDistributionBucket.count
DatasetDistributionDistributionBucket.left
DatasetDistributionDistributionBucket.right
DatasetDistributionDistributionBucketDict
DatasetDistributionDistributionBucketDict.count
DatasetDistributionDistributionBucketDict.left
DatasetDistributionDistributionBucketDict.right
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 64/152
```
DatasetStats
DatasetStats.total_billable_character_count
DatasetStats.total_tuning_character_count
DatasetStats.tuning_dataset_example_count
DatasetStats.tuning_step_count
DatasetStats.user_dataset_examples
DatasetStats.user_input_token_distribution
DatasetStats.user_message_per_example_distribution
DatasetStats.user_output_token_distribution
DatasetStatsDict
DatasetStatsDict.total_billable_character_count
DatasetStatsDict.total_tuning_character_count
DatasetStatsDict.tuning_dataset_example_count
DatasetStatsDict.tuning_step_count
DatasetStatsDict.user_dataset_examples
DatasetStatsDict.user_input_token_distribution
DatasetStatsDict.user_message_per_example_distribution
DatasetStatsDict.user_output_token_distribution
DeleteBatchJobConfig
DeleteBatchJobConfig.http_options
DeleteBatchJobConfigDict
DeleteBatchJobConfigDict.http_options
DeleteCachedContentConfig
DeleteCachedContentConfig.http_options
DeleteCachedContentConfigDict
DeleteCachedContentConfigDict.http_options
DeleteCachedContentResponse
DeleteCachedContentResponse.sdk_http_response
DeleteCachedContentResponseDict
DeleteCachedContentResponseDict.sdk_http_response
DeleteDocumentConfig
DeleteDocumentConfig.force
DeleteDocumentConfig.http_options
DeleteDocumentConfigDict
DeleteDocumentConfigDict.force
DeleteDocumentConfigDict.http_options
DeleteFileConfig
DeleteFileConfig.http_options
DeleteFileConfigDict
DeleteFileConfigDict.http_options
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 65/152
```
DeleteFileResponse
DeleteFileResponse.sdk_http_response
DeleteFileResponseDict
DeleteFileResponseDict.sdk_http_response
DeleteFileSearchStoreConfig
DeleteFileSearchStoreConfig.force
DeleteFileSearchStoreConfig.http_options
DeleteFileSearchStoreConfigDict
DeleteFileSearchStoreConfigDict.force
DeleteFileSearchStoreConfigDict.http_options
DeleteModelConfig
DeleteModelConfig.http_options
DeleteModelConfigDict
DeleteModelConfigDict.http_options
DeleteModelResponse
DeleteModelResponse.sdk_http_response
DeleteModelResponseDict
DeleteModelResponseDict.sdk_http_response
DeleteResourceJob
DeleteResourceJob.done
DeleteResourceJob.error
DeleteResourceJob.name
DeleteResourceJob.sdk_http_response
DeleteResourceJobDict
DeleteResourceJobDict.done
DeleteResourceJobDict.error
DeleteResourceJobDict.name
DeleteResourceJobDict.sdk_http_response
DistillationDataStats
DistillationDataStats.training_dataset_stats
DistillationDataStatsDict
DistillationDataStatsDict.training_dataset_stats
DistillationHyperParameters
DistillationHyperParameters.adapter_size
DistillationHyperParameters.epoch_count
DistillationHyperParameters.learning_rate_multiplier
DistillationHyperParametersDict
DistillationHyperParametersDict.adapter_size
DistillationHyperParametersDict.epoch_count
DistillationHyperParametersDict.learning_rate_multiplier
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 66/152
```
DistillationSpec
DistillationSpec.base_teacher_model
DistillationSpec.hyper_parameters
DistillationSpec.pipeline_root_directory
DistillationSpec.prompt_dataset_uri
DistillationSpec.student_model
DistillationSpec.training_dataset_uri
DistillationSpec.tuned_teacher_model_source
DistillationSpec.validation_dataset_uri
DistillationSpecDict
DistillationSpecDict.base_teacher_model
DistillationSpecDict.hyper_parameters
DistillationSpecDict.pipeline_root_directory
DistillationSpecDict.prompt_dataset_uri
DistillationSpecDict.student_model
DistillationSpecDict.training_dataset_uri
DistillationSpecDict.tuned_teacher_model_source
DistillationSpecDict.validation_dataset_uri
Document
Document.create_time
Document.custom_metadata
Document.display_name
Document.mime_type
Document.name
Document.size_bytes
Document.state
Document.update_time
DocumentDict
DocumentDict.create_time
DocumentDict.custom_metadata
DocumentDict.display_name
DocumentDict.mime_type
DocumentDict.name
DocumentDict.size_bytes
DocumentDict.state
DocumentDict.update_time
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 67/152
```
DocumentState
DocumentState.STATE_ACTIVE
DocumentState.STATE_FAILED
DocumentState.STATE_PENDING
DocumentState.STATE_UNSPECIFIED
DownloadFileConfig
DownloadFileConfig.http_options
DownloadFileConfigDict
DownloadFileConfigDict.http_options
DynamicRetrievalConfig
DynamicRetrievalConfig.dynamic_threshold
DynamicRetrievalConfig.mode
DynamicRetrievalConfigDict
DynamicRetrievalConfigDict.dynamic_threshold
DynamicRetrievalConfigDict.mode
DynamicRetrievalConfigMode
DynamicRetrievalConfigMode.MODE_DYNAMIC
DynamicRetrievalConfigMode.MODE_UNSPECIFIED
EditImageConfig
EditImageConfig.add_watermark
EditImageConfig.aspect_ratio
EditImageConfig.base_steps
EditImageConfig.edit_mode
EditImageConfig.guidance_scale
EditImageConfig.http_options
EditImageConfig.include_rai_reason
EditImageConfig.include_safety_attributes
EditImageConfig.labels
EditImageConfig.language
EditImageConfig.negative_prompt
EditImageConfig.number_of_images
EditImageConfig.output_compression_quality
EditImageConfig.output_gcs_uri
EditImageConfig.output_mime_type
EditImageConfig.person_generation
EditImageConfig.safety_filter_level
EditImageConfig.seed
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 68/152
```
EditImageConfigDict
EditImageConfigDict.add_watermark
EditImageConfigDict.aspect_ratio
EditImageConfigDict.base_steps
EditImageConfigDict.edit_mode
EditImageConfigDict.guidance_scale
EditImageConfigDict.http_options
EditImageConfigDict.include_rai_reason
EditImageConfigDict.include_safety_attributes
EditImageConfigDict.labels
EditImageConfigDict.language
EditImageConfigDict.negative_prompt
EditImageConfigDict.number_of_images
EditImageConfigDict.output_compression_quality
EditImageConfigDict.output_gcs_uri
EditImageConfigDict.output_mime_type
EditImageConfigDict.person_generation
EditImageConfigDict.safety_filter_level
EditImageConfigDict.seed
EditImageResponse
EditImageResponse.generated_images
EditImageResponse.sdk_http_response
EditImageResponseDict
EditImageResponseDict.generated_images
EditImageResponseDict.sdk_http_response
EditMode
EditMode.EDIT_MODE_BGSWAP
EditMode.EDIT_MODE_CONTROLLED_EDITING
EditMode.EDIT_MODE_DEFAULT
EditMode.EDIT_MODE_INPAINT_INSERTION
EditMode.EDIT_MODE_INPAINT_REMOVAL
EditMode.EDIT_MODE_OUTPAINT
EditMode.EDIT_MODE_PRODUCT_IMAGE
EditMode.EDIT_MODE_STYLE
EmbedContentBatch
EmbedContentBatch.config
EmbedContentBatch.contents
EmbedContentBatchDict
EmbedContentBatchDict.config
EmbedContentBatchDict.contents
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 69/152
```
EmbedContentConfig
EmbedContentConfig.auto_truncate
EmbedContentConfig.http_options
EmbedContentConfig.mime_type
EmbedContentConfig.output_dimensionality
EmbedContentConfig.task_type
EmbedContentConfig.title
EmbedContentConfigDict
EmbedContentConfigDict.auto_truncate
EmbedContentConfigDict.http_options
EmbedContentConfigDict.mime_type
EmbedContentConfigDict.output_dimensionality
EmbedContentConfigDict.task_type
EmbedContentConfigDict.title
EmbedContentMetadata
EmbedContentMetadata.billable_character_count
EmbedContentMetadataDict
EmbedContentMetadataDict.billable_character_count
EmbedContentParameters
EmbedContentParameters.config
EmbedContentParameters.contents
EmbedContentParameters.model
EmbedContentParametersDict
EmbedContentParametersDict.config
EmbedContentParametersDict.contents
EmbedContentParametersDict.model
EmbedContentResponse
EmbedContentResponse.embeddings
EmbedContentResponse.metadata
EmbedContentResponse.sdk_http_response
EmbedContentResponseDict
EmbedContentResponseDict.embeddings
EmbedContentResponseDict.metadata
EmbedContentResponseDict.sdk_http_response
EmbeddingApiType
EmbeddingApiType.EMBED_CONTENT
EmbeddingApiType.PREDICT
EmbeddingsBatchJobSource
EmbeddingsBatchJobSource.file_name
EmbeddingsBatchJobSource.inlined_requests
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 70/152
```
EmbeddingsBatchJobSourceDict
EmbeddingsBatchJobSourceDict.file_name
EmbeddingsBatchJobSourceDict.inlined_requests
EncryptionSpec
EncryptionSpec.kms_key_name
EncryptionSpecDict
EncryptionSpecDict.kms_key_name
EndSensitivity
EndSensitivity.END_SENSITIVITY_HIGH
EndSensitivity.END_SENSITIVITY_LOW
EndSensitivity.END_SENSITIVITY_UNSPECIFIED
Endpoint
Endpoint.deployed_model_id
Endpoint.name
EndpointDict
EndpointDict.deployed_model_id
EndpointDict.name
EnterpriseWebSearch
EnterpriseWebSearch.blocking_confidence
EnterpriseWebSearch.exclude_domains
EnterpriseWebSearchDict
EnterpriseWebSearchDict.blocking_confidence
EnterpriseWebSearchDict.exclude_domains
EntityLabel
EntityLabel.label
EntityLabel.score
EntityLabelDict
EntityLabelDict.label
EntityLabelDict.score
Environment
Environment.ENVIRONMENT_BROWSER
Environment.ENVIRONMENT_UNSPECIFIED
EvaluationConfig
EvaluationConfig.autorater_config
EvaluationConfig.metrics
EvaluationConfig.output_config
EvaluationConfigDict
EvaluationConfigDict.autorater_config
EvaluationConfigDict.metrics
EvaluationConfigDict.output_config
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 71/152
```
ExecutableCode
ExecutableCode.code
ExecutableCode.language
ExecutableCodeDict
ExecutableCodeDict.code
ExecutableCodeDict.language
ExternalApi
ExternalApi.api_auth
ExternalApi.api_spec
ExternalApi.auth_config
ExternalApi.elastic_search_params
ExternalApi.endpoint
ExternalApi.simple_search_params
ExternalApiDict
ExternalApiDict.api_auth
ExternalApiDict.api_spec
ExternalApiDict.auth_config
ExternalApiDict.elastic_search_params
ExternalApiDict.endpoint
ExternalApiDict.simple_search_params
ExternalApiElasticSearchParams
ExternalApiElasticSearchParams.index
ExternalApiElasticSearchParams.num_hits
ExternalApiElasticSearchParams.search_template
ExternalApiElasticSearchParamsDict
ExternalApiElasticSearchParamsDict.index
ExternalApiElasticSearchParamsDict.num_hits
ExternalApiElasticSearchParamsDict.search_template
ExternalApiSimpleSearchParams
ExternalApiSimpleSearchParamsDict
FeatureSelectionPreference
FeatureSelectionPreference.BALANCED
FeatureSelectionPreference.FEATURE_SELECTION_PREFERENCE_UNSPECIFIED
FeatureSelectionPreference.PRIORITIZE_COST
FeatureSelectionPreference.PRIORITIZE_QUALITY
FetchPredictOperationConfig
FetchPredictOperationConfig.http_options
FetchPredictOperationConfigDict
FetchPredictOperationConfigDict.http_options
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 72/152
```
File
File.create_time
File.display_name
File.download_uri
File.error
File.expiration_time
File.mime_type
File.name
File.sha256_hash
File.size_bytes
File.source
File.state
File.update_time
File.uri
File.video_metadata
FileData
FileData.display_name
FileData.file_uri
FileData.mime_type
FileDataDict
FileDataDict.display_name
FileDataDict.file_uri
FileDataDict.mime_type
FileDict
FileDict.create_time
FileDict.display_name
FileDict.download_uri
FileDict.error
FileDict.expiration_time
FileDict.mime_type
FileDict.name
FileDict.sha256_hash
FileDict.size_bytes
FileDict.source
FileDict.state
FileDict.update_time
FileDict.uri
FileDict.video_metadataSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 73/152
```
FileSearch
FileSearch.file_search_store_names
FileSearch.metadata_filter
FileSearch.top_k
FileSearchDict
FileSearchDict.file_search_store_names
FileSearchDict.metadata_filter
FileSearchDict.top_k
FileSearchStore
FileSearchStore.active_documents_count
FileSearchStore.create_time
FileSearchStore.display_name
FileSearchStore.failed_documents_count
FileSearchStore.name
FileSearchStore.pending_documents_count
FileSearchStore.size_bytes
FileSearchStore.update_time
FileSearchStoreDict
FileSearchStoreDict.active_documents_count
FileSearchStoreDict.create_time
FileSearchStoreDict.display_name
FileSearchStoreDict.failed_documents_count
FileSearchStoreDict.name
FileSearchStoreDict.pending_documents_count
FileSearchStoreDict.size_bytes
FileSearchStoreDict.update_time
FileSource
FileSource.GENERATED
FileSource.REGISTERED
FileSource.SOURCE_UNSPECIFIED
FileSource.UPLOADED
FileState
FileState.ACTIVE
FileState.FAILED
FileState.PROCESSING
FileState.STATE_UNSPECIFIED
FileStatus
FileStatus.code
FileStatus.details
FileStatus.message
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 74/152
```
FileStatusDict
FileStatusDict.code
FileStatusDict.details
FileStatusDict.message
FinishReason
FinishReason.BLOCKLIST
FinishReason.FINISH_REASON_UNSPECIFIED
FinishReason.IMAGE_OTHER
FinishReason.IMAGE_PROHIBITED_CONTENT
FinishReason.IMAGE_RECITATION
FinishReason.IMAGE_SAFETY
FinishReason.LANGUAGE
FinishReason.MALFORMED_FUNCTION_CALL
FinishReason.MAX_TOKENS
FinishReason.NO_IMAGE
FinishReason.OTHER
FinishReason.PROHIBITED_CONTENT
FinishReason.RECITATION
FinishReason.SAFETY
FinishReason.SPII
FinishReason.STOP
FinishReason.UNEXPECTED_TOOL_CALL
FunctionCall
FunctionCall.args
FunctionCall.id
FunctionCall.name
FunctionCall.partial_args
FunctionCall.will_continue
FunctionCallDict
FunctionCallDict.args
FunctionCallDict.id
FunctionCallDict.name
FunctionCallDict.partial_args
FunctionCallDict.will_continue
FunctionCallingConfig
FunctionCallingConfig.allowed_function_names
FunctionCallingConfig.mode
FunctionCallingConfig.stream_function_call_argumentsSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 75/152
```
FunctionCallingConfigDict
FunctionCallingConfigDict.allowed_function_names
FunctionCallingConfigDict.mode
FunctionCallingConfigDict.stream_function_call_arguments
FunctionCallingConfigMode
FunctionCallingConfigMode.ANY
FunctionCallingConfigMode.AUTO
FunctionCallingConfigMode.MODE_UNSPECIFIED
FunctionCallingConfigMode.NONE
FunctionCallingConfigMode.VALIDATED
FunctionDeclaration
FunctionDeclaration.behavior
FunctionDeclaration.description
FunctionDeclaration.name
FunctionDeclaration.parameters
FunctionDeclaration.parameters_json_schema
FunctionDeclaration.response
FunctionDeclaration.response_json_schema
```
FunctionDeclaration.from_callable()
```
```
FunctionDeclaration.from_callable_with_api_option()
```
FunctionDeclarationDict
FunctionDeclarationDict.behavior
FunctionDeclarationDict.description
FunctionDeclarationDict.name
FunctionDeclarationDict.parameters
FunctionDeclarationDict.parameters_json_schema
FunctionDeclarationDict.response
FunctionDeclarationDict.response_json_schema
FunctionResponse
FunctionResponse.id
FunctionResponse.name
FunctionResponse.parts
FunctionResponse.response
FunctionResponse.scheduling
FunctionResponse.will_continue
```
FunctionResponse.from_mcp_response()
```
FunctionResponseBlob
FunctionResponseBlob.data
FunctionResponseBlob.display_name
FunctionResponseBlob.mime_type
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 76/152
```
FunctionResponseBlobDict
FunctionResponseBlobDict.data
FunctionResponseBlobDict.display_name
FunctionResponseBlobDict.mime_type
FunctionResponseDict
FunctionResponseDict.id
FunctionResponseDict.name
FunctionResponseDict.parts
FunctionResponseDict.response
FunctionResponseDict.scheduling
FunctionResponseDict.will_continue
FunctionResponseFileData
FunctionResponseFileData.display_name
FunctionResponseFileData.file_uri
FunctionResponseFileData.mime_type
FunctionResponseFileDataDict
FunctionResponseFileDataDict.display_name
FunctionResponseFileDataDict.file_uri
FunctionResponseFileDataDict.mime_type
FunctionResponsePart
FunctionResponsePart.file_data
FunctionResponsePart.inline_data
```
FunctionResponsePart.from_bytes()
```
```
FunctionResponsePart.from_uri()
```
FunctionResponsePartDict
FunctionResponsePartDict.file_data
FunctionResponsePartDict.inline_data
FunctionResponseScheduling
FunctionResponseScheduling.INTERRUPT
FunctionResponseScheduling.SCHEDULING_UNSPECIFIED
FunctionResponseScheduling.SILENT
FunctionResponseScheduling.WHEN_IDLE
GcsDestination
GcsDestination.output_uri_prefix
GcsDestinationDict
GcsDestinationDict.output_uri_prefix
GeminiPreferenceExample
GeminiPreferenceExample.completions
GeminiPreferenceExample.contents
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 77/152
```
GeminiPreferenceExampleCompletion
GeminiPreferenceExampleCompletion.completion
GeminiPreferenceExampleCompletion.score
GeminiPreferenceExampleCompletionDict
GeminiPreferenceExampleCompletionDict.completion
GeminiPreferenceExampleCompletionDict.score
GeminiPreferenceExampleDict
GeminiPreferenceExampleDict.completions
GeminiPreferenceExampleDict.contents
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 78/152
```
GenerateContentConfig
GenerateContentConfig.audio_timestamp
GenerateContentConfig.automatic_function_calling
GenerateContentConfig.cached_content
GenerateContentConfig.candidate_count
GenerateContentConfig.enable_enhanced_civic_answers
GenerateContentConfig.frequency_penalty
GenerateContentConfig.http_options
GenerateContentConfig.image_config
GenerateContentConfig.labels
GenerateContentConfig.logprobs
GenerateContentConfig.max_output_tokens
GenerateContentConfig.media_resolution
GenerateContentConfig.model_armor_config
GenerateContentConfig.model_selection_config
GenerateContentConfig.presence_penalty
GenerateContentConfig.response_json_schema
GenerateContentConfig.response_logprobs
GenerateContentConfig.response_mime_type
GenerateContentConfig.response_modalities
GenerateContentConfig.response_schema
GenerateContentConfig.routing_config
GenerateContentConfig.safety_settings
GenerateContentConfig.seed
GenerateContentConfig.should_return_http_response
GenerateContentConfig.speech_config
GenerateContentConfig.stop_sequences
GenerateContentConfig.system_instruction
GenerateContentConfig.temperature
GenerateContentConfig.thinking_config
GenerateContentConfig.tool_config
GenerateContentConfig.tools
GenerateContentConfig.top_k
GenerateContentConfig.top_p
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 79/152
```
GenerateContentConfigDict
GenerateContentConfigDict.audio_timestamp
GenerateContentConfigDict.automatic_function_calling
GenerateContentConfigDict.cached_content
GenerateContentConfigDict.candidate_count
GenerateContentConfigDict.enable_enhanced_civic_answers
GenerateContentConfigDict.frequency_penalty
GenerateContentConfigDict.http_options
GenerateContentConfigDict.image_config
GenerateContentConfigDict.labels
GenerateContentConfigDict.logprobs
GenerateContentConfigDict.max_output_tokens
GenerateContentConfigDict.media_resolution
GenerateContentConfigDict.model_armor_config
GenerateContentConfigDict.model_selection_config
GenerateContentConfigDict.presence_penalty
GenerateContentConfigDict.response_json_schema
GenerateContentConfigDict.response_logprobs
GenerateContentConfigDict.response_mime_type
GenerateContentConfigDict.response_modalities
GenerateContentConfigDict.response_schema
GenerateContentConfigDict.routing_config
GenerateContentConfigDict.safety_settings
GenerateContentConfigDict.seed
GenerateContentConfigDict.should_return_http_response
GenerateContentConfigDict.speech_config
GenerateContentConfigDict.stop_sequences
GenerateContentConfigDict.system_instruction
GenerateContentConfigDict.temperature
GenerateContentConfigDict.thinking_config
GenerateContentConfigDict.tool_config
GenerateContentConfigDict.tools
GenerateContentConfigDict.top_k
GenerateContentConfigDict.top_p
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 80/152
```
GenerateContentResponse
GenerateContentResponse.automatic_function_calling_history
GenerateContentResponse.candidates
GenerateContentResponse.create_time
GenerateContentResponse.model_version
GenerateContentResponse.parsed
GenerateContentResponse.prompt_feedback
GenerateContentResponse.response_id
GenerateContentResponse.sdk_http_response
GenerateContentResponse.usage_metadata
GenerateContentResponse.code_execution_result
GenerateContentResponse.executable_code
GenerateContentResponse.function_calls
GenerateContentResponse.parts
GenerateContentResponse.text
GenerateContentResponseDict
GenerateContentResponseDict.candidates
GenerateContentResponseDict.create_time
GenerateContentResponseDict.model_version
GenerateContentResponseDict.prompt_feedback
GenerateContentResponseDict.response_id
GenerateContentResponseDict.sdk_http_response
GenerateContentResponseDict.usage_metadata
GenerateContentResponsePromptFeedback
GenerateContentResponsePromptFeedback.block_reason
GenerateContentResponsePromptFeedback.block_reason_message
GenerateContentResponsePromptFeedback.safety_ratings
GenerateContentResponsePromptFeedbackDict
GenerateContentResponsePromptFeedbackDict.block_reason
GenerateContentResponsePromptFeedbackDict.block_reason_message
GenerateContentResponsePromptFeedbackDict.safety_ratings
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 81/152
```
GenerateContentResponseUsageMetadata
GenerateContentResponseUsageMetadata.cache_tokens_details
GenerateContentResponseUsageMetadata.cached_content_token_count
GenerateContentResponseUsageMetadata.candidates_token_count
GenerateContentResponseUsageMetadata.candidates_tokens_details
GenerateContentResponseUsageMetadata.prompt_token_count
GenerateContentResponseUsageMetadata.prompt_tokens_details
GenerateContentResponseUsageMetadata.thoughts_token_count
GenerateContentResponseUsageMetadata.tool_use_prompt_token_count
GenerateContentResponseUsageMetadata.tool_use_prompt_tokens_details
GenerateContentResponseUsageMetadata.total_token_count
GenerateContentResponseUsageMetadata.traffic_type
GenerateContentResponseUsageMetadataDict
GenerateContentResponseUsageMetadataDict.cache_tokens_details
GenerateContentResponseUsageMetadataDict.cached_content_token_count
GenerateContentResponseUsageMetadataDict.candidates_token_count
GenerateContentResponseUsageMetadataDict.candidates_tokens_details
GenerateContentResponseUsageMetadataDict.prompt_token_count
GenerateContentResponseUsageMetadataDict.prompt_tokens_details
GenerateContentResponseUsageMetadataDict.thoughts_token_count
GenerateContentResponseUsageMetadataDict.tool_use_prompt_token_count
GenerateContentResponseUsageMetadataDict.tool_use_prompt_tokens_details
GenerateContentResponseUsageMetadataDict.total_token_count
GenerateContentResponseUsageMetadataDict.traffic_type
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 82/152
```
GenerateImagesConfig
GenerateImagesConfig.add_watermark
GenerateImagesConfig.aspect_ratio
GenerateImagesConfig.enhance_prompt
GenerateImagesConfig.guidance_scale
GenerateImagesConfig.http_options
GenerateImagesConfig.image_size
GenerateImagesConfig.include_rai_reason
GenerateImagesConfig.include_safety_attributes
GenerateImagesConfig.labels
GenerateImagesConfig.language
GenerateImagesConfig.negative_prompt
GenerateImagesConfig.number_of_images
GenerateImagesConfig.output_compression_quality
GenerateImagesConfig.output_gcs_uri
GenerateImagesConfig.output_mime_type
GenerateImagesConfig.person_generation
GenerateImagesConfig.safety_filter_level
GenerateImagesConfig.seed
GenerateImagesConfigDict
GenerateImagesConfigDict.add_watermark
GenerateImagesConfigDict.aspect_ratio
GenerateImagesConfigDict.enhance_prompt
GenerateImagesConfigDict.guidance_scale
GenerateImagesConfigDict.http_options
GenerateImagesConfigDict.image_size
GenerateImagesConfigDict.include_rai_reason
GenerateImagesConfigDict.include_safety_attributes
GenerateImagesConfigDict.labels
GenerateImagesConfigDict.language
GenerateImagesConfigDict.negative_prompt
GenerateImagesConfigDict.number_of_images
GenerateImagesConfigDict.output_compression_quality
GenerateImagesConfigDict.output_gcs_uri
GenerateImagesConfigDict.output_mime_type
GenerateImagesConfigDict.person_generation
GenerateImagesConfigDict.safety_filter_level
GenerateImagesConfigDict.seedSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 83/152
```
GenerateImagesResponse
GenerateImagesResponse.generated_images
GenerateImagesResponse.positive_prompt_safety_attributes
GenerateImagesResponse.sdk_http_response
GenerateImagesResponse.images
GenerateImagesResponseDict
GenerateImagesResponseDict.generated_images
GenerateImagesResponseDict.positive_prompt_safety_attributes
GenerateImagesResponseDict.sdk_http_response
GenerateVideosConfig
GenerateVideosConfig.aspect_ratio
GenerateVideosConfig.compression_quality
GenerateVideosConfig.duration_seconds
GenerateVideosConfig.enhance_prompt
GenerateVideosConfig.fps
GenerateVideosConfig.generate_audio
GenerateVideosConfig.http_options
GenerateVideosConfig.last_frame
GenerateVideosConfig.mask
GenerateVideosConfig.negative_prompt
GenerateVideosConfig.number_of_videos
GenerateVideosConfig.output_gcs_uri
GenerateVideosConfig.person_generation
GenerateVideosConfig.pubsub_topic
GenerateVideosConfig.reference_images
GenerateVideosConfig.resolution
GenerateVideosConfig.seed
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 84/152
```
GenerateVideosConfigDict
GenerateVideosConfigDict.aspect_ratio
GenerateVideosConfigDict.compression_quality
GenerateVideosConfigDict.duration_seconds
GenerateVideosConfigDict.enhance_prompt
GenerateVideosConfigDict.fps
GenerateVideosConfigDict.generate_audio
GenerateVideosConfigDict.http_options
GenerateVideosConfigDict.last_frame
GenerateVideosConfigDict.mask
GenerateVideosConfigDict.negative_prompt
GenerateVideosConfigDict.number_of_videos
GenerateVideosConfigDict.output_gcs_uri
GenerateVideosConfigDict.person_generation
GenerateVideosConfigDict.pubsub_topic
GenerateVideosConfigDict.reference_images
GenerateVideosConfigDict.resolution
GenerateVideosConfigDict.seed
GenerateVideosOperation
GenerateVideosOperation.response
GenerateVideosOperation.result
```
GenerateVideosOperation.from_api_response()
```
GenerateVideosResponse
GenerateVideosResponse.generated_videos
GenerateVideosResponse.rai_media_filtered_count
GenerateVideosResponse.rai_media_filtered_reasons
GenerateVideosResponseDict
GenerateVideosResponseDict.generated_videos
GenerateVideosResponseDict.rai_media_filtered_count
GenerateVideosResponseDict.rai_media_filtered_reasons
GenerateVideosSource
GenerateVideosSource.image
GenerateVideosSource.prompt
GenerateVideosSource.video
GenerateVideosSourceDict
GenerateVideosSourceDict.image
GenerateVideosSourceDict.prompt
GenerateVideosSourceDict.videoSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 85/152
```
GeneratedImage
GeneratedImage.enhanced_prompt
GeneratedImage.image
GeneratedImage.rai_filtered_reason
GeneratedImage.safety_attributes
GeneratedImageDict
GeneratedImageDict.enhanced_prompt
GeneratedImageDict.image
GeneratedImageDict.rai_filtered_reason
GeneratedImageDict.safety_attributes
GeneratedImageMask
GeneratedImageMask.labels
GeneratedImageMask.mask
GeneratedImageMaskDict
GeneratedImageMaskDict.labels
GeneratedImageMaskDict.mask
GeneratedVideo
GeneratedVideo.video
GeneratedVideoDict
GeneratedVideoDict.video
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 86/152
```
GenerationConfig
GenerationConfig.audio_timestamp
GenerationConfig.candidate_count
GenerationConfig.enable_affective_dialog
GenerationConfig.enable_enhanced_civic_answers
GenerationConfig.frequency_penalty
GenerationConfig.logprobs
GenerationConfig.max_output_tokens
GenerationConfig.media_resolution
GenerationConfig.model_selection_config
GenerationConfig.presence_penalty
GenerationConfig.response_json_schema
GenerationConfig.response_logprobs
GenerationConfig.response_mime_type
GenerationConfig.response_modalities
GenerationConfig.response_schema
GenerationConfig.routing_config
GenerationConfig.seed
GenerationConfig.speech_config
GenerationConfig.stop_sequences
GenerationConfig.temperature
GenerationConfig.thinking_config
GenerationConfig.top_k
GenerationConfig.top_p
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 87/152
```
GenerationConfigDict
GenerationConfigDict.audio_timestamp
GenerationConfigDict.candidate_count
GenerationConfigDict.enable_affective_dialog
GenerationConfigDict.enable_enhanced_civic_answers
GenerationConfigDict.frequency_penalty
GenerationConfigDict.logprobs
GenerationConfigDict.max_output_tokens
GenerationConfigDict.media_resolution
GenerationConfigDict.model_selection_config
GenerationConfigDict.presence_penalty
GenerationConfigDict.response_json_schema
GenerationConfigDict.response_logprobs
GenerationConfigDict.response_mime_type
GenerationConfigDict.response_modalities
GenerationConfigDict.response_schema
GenerationConfigDict.routing_config
GenerationConfigDict.seed
GenerationConfigDict.speech_config
GenerationConfigDict.stop_sequences
GenerationConfigDict.temperature
GenerationConfigDict.thinking_config
GenerationConfigDict.top_k
GenerationConfigDict.top_p
GenerationConfigRoutingConfig
GenerationConfigRoutingConfig.auto_mode
GenerationConfigRoutingConfig.manual_mode
GenerationConfigRoutingConfigAutoRoutingMode
GenerationConfigRoutingConfigAutoRoutingMode.model_routing_preference
GenerationConfigRoutingConfigAutoRoutingModeDict
GenerationConfigRoutingConfigAutoRoutingModeDict.model_routing_preference
GenerationConfigRoutingConfigDict
GenerationConfigRoutingConfigDict.auto_mode
GenerationConfigRoutingConfigDict.manual_mode
GenerationConfigRoutingConfigManualRoutingMode
GenerationConfigRoutingConfigManualRoutingMode.model_name
GenerationConfigRoutingConfigManualRoutingModeDict
GenerationConfigRoutingConfigManualRoutingModeDict.model_name
GenerationConfigThinkingConfig
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 88/152
```
GenerationConfigThinkingConfigDict
GenerationConfigThinkingConfigDict.include_thoughts
GenerationConfigThinkingConfigDict.thinking_budget
GenerationConfigThinkingConfigDict.thinking_level
GetBatchJobConfig
GetBatchJobConfig.http_options
GetBatchJobConfigDict
GetBatchJobConfigDict.http_options
GetCachedContentConfig
GetCachedContentConfig.http_options
GetCachedContentConfigDict
GetCachedContentConfigDict.http_options
GetDocumentConfig
GetDocumentConfig.http_options
GetDocumentConfigDict
GetDocumentConfigDict.http_options
GetFileConfig
GetFileConfig.http_options
GetFileConfigDict
GetFileConfigDict.http_options
GetFileSearchStoreConfig
GetFileSearchStoreConfig.http_options
GetFileSearchStoreConfigDict
GetFileSearchStoreConfigDict.http_options
GetModelConfig
GetModelConfig.http_options
GetModelConfigDict
GetModelConfigDict.http_options
GetOperationConfig
GetOperationConfig.http_options
GetOperationConfigDict
GetOperationConfigDict.http_options
GetTuningJobConfig
GetTuningJobConfig.http_options
GetTuningJobConfigDict
GetTuningJobConfigDict.http_options
GoogleMaps
GoogleMaps.auth_config
GoogleMaps.enable_widget
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 89/152
```
GoogleMapsDict
GoogleMapsDict.auth_config
GoogleMapsDict.enable_widget
GoogleRpcStatus
GoogleRpcStatus.code
GoogleRpcStatus.details
GoogleRpcStatus.message
GoogleRpcStatusDict
GoogleRpcStatusDict.code
GoogleRpcStatusDict.details
GoogleRpcStatusDict.message
GoogleSearch
GoogleSearch.blocking_confidence
GoogleSearch.exclude_domains
GoogleSearch.search_types
GoogleSearch.time_range_filter
GoogleSearchDict
GoogleSearchDict.blocking_confidence
GoogleSearchDict.exclude_domains
GoogleSearchDict.search_types
GoogleSearchDict.time_range_filter
GoogleSearchRetrieval
GoogleSearchRetrieval.dynamic_retrieval_config
GoogleSearchRetrievalDict
GoogleSearchRetrievalDict.dynamic_retrieval_config
GoogleTypeDate
GoogleTypeDate.day
GoogleTypeDate.month
GoogleTypeDate.year
GoogleTypeDateDict
GoogleTypeDateDict.day
GoogleTypeDateDict.month
GoogleTypeDateDict.year
GroundingChunk
GroundingChunk.image
GroundingChunk.maps
GroundingChunk.retrieved_context
GroundingChunk.webSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 90/152
```
GroundingChunkDict
GroundingChunkDict.image
GroundingChunkDict.maps
GroundingChunkDict.retrieved_context
GroundingChunkDict.web
GroundingChunkImage
GroundingChunkImage.domain
GroundingChunkImage.image_uri
GroundingChunkImage.source_uri
GroundingChunkImage.title
GroundingChunkImageDict
GroundingChunkImageDict.domain
GroundingChunkImageDict.image_uri
GroundingChunkImageDict.source_uri
GroundingChunkImageDict.title
GroundingChunkMaps
GroundingChunkMaps.place_answer_sources
GroundingChunkMaps.place_id
GroundingChunkMaps.text
GroundingChunkMaps.title
GroundingChunkMaps.uri
GroundingChunkMapsDict
GroundingChunkMapsDict.place_answer_sources
GroundingChunkMapsDict.place_id
GroundingChunkMapsDict.text
GroundingChunkMapsDict.title
GroundingChunkMapsDict.uri
GroundingChunkMapsPlaceAnswerSources
GroundingChunkMapsPlaceAnswerSources.flag_content_uri
GroundingChunkMapsPlaceAnswerSources.review_snippets
GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution
GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution.display_name
GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution.photo_uri
GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution.uri
GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict
GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict.display_name
GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict.photo_uri
GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict.uriSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 91/152
```
GroundingChunkMapsPlaceAnswerSourcesDict
GroundingChunkMapsPlaceAnswerSourcesDict.flag_content_uri
GroundingChunkMapsPlaceAnswerSourcesDict.review_snippets
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.author_attribution
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.flag_content_uri
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.google_maps_uri
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.relative_publish_time_descriptio
n
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.review
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.review_id
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet.title
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.author_attribution
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.flag_content_uri
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.google_maps_uri
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.relative_publish_time_descrip
tion
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.review
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.review_id
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict.title
GroundingChunkRetrievedContext
GroundingChunkRetrievedContext.document_name
GroundingChunkRetrievedContext.rag_chunk
GroundingChunkRetrievedContext.text
GroundingChunkRetrievedContext.title
GroundingChunkRetrievedContext.uri
GroundingChunkRetrievedContextDict
GroundingChunkRetrievedContextDict.document_name
GroundingChunkRetrievedContextDict.rag_chunk
GroundingChunkRetrievedContextDict.text
GroundingChunkRetrievedContextDict.title
GroundingChunkRetrievedContextDict.uri
GroundingChunkWeb
GroundingChunkWeb.domain
GroundingChunkWeb.title
GroundingChunkWeb.uri
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 92/152
```
GroundingChunkWebDict
GroundingChunkWebDict.domain
GroundingChunkWebDict.title
GroundingChunkWebDict.uri
GroundingMetadata
GroundingMetadata.google_maps_widget_context_token
GroundingMetadata.grounding_chunks
GroundingMetadata.grounding_supports
GroundingMetadata.image_search_queries
GroundingMetadata.retrieval_metadata
GroundingMetadata.retrieval_queries
GroundingMetadata.search_entry_point
GroundingMetadata.source_flagging_uris
GroundingMetadata.web_search_queries
GroundingMetadataDict
GroundingMetadataDict.google_maps_widget_context_token
GroundingMetadataDict.grounding_chunks
GroundingMetadataDict.grounding_supports
GroundingMetadataDict.image_search_queries
GroundingMetadataDict.retrieval_metadata
GroundingMetadataDict.retrieval_queries
GroundingMetadataDict.search_entry_point
GroundingMetadataDict.source_flagging_uris
GroundingMetadataDict.web_search_queries
GroundingMetadataSourceFlaggingUri
GroundingMetadataSourceFlaggingUri.flag_content_uri
GroundingMetadataSourceFlaggingUri.source_id
GroundingMetadataSourceFlaggingUriDict
GroundingMetadataSourceFlaggingUriDict.flag_content_uri
GroundingMetadataSourceFlaggingUriDict.source_id
GroundingSupport
GroundingSupport.confidence_scores
GroundingSupport.grounding_chunk_indices
GroundingSupport.segment
GroundingSupportDict
GroundingSupportDict.confidence_scores
GroundingSupportDict.grounding_chunk_indices
GroundingSupportDict.segmentSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 93/152
```
HarmBlockMethod
HarmBlockMethod.HARM_BLOCK_METHOD_UNSPECIFIED
HarmBlockMethod.PROBABILITY
HarmBlockMethod.SEVERITY
HarmBlockThreshold
HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
HarmBlockThreshold.BLOCK_NONE
HarmBlockThreshold.BLOCK_ONLY_HIGH
HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED
HarmBlockThreshold.OFF
HarmCategory
HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY
HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
HarmCategory.HARM_CATEGORY_HARASSMENT
HarmCategory.HARM_CATEGORY_HATE_SPEECH
HarmCategory.HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT
HarmCategory.HARM_CATEGORY_IMAGE_HARASSMENT
HarmCategory.HARM_CATEGORY_IMAGE_HATE
HarmCategory.HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT
HarmCategory.HARM_CATEGORY_JAILBREAK
HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
HarmCategory.HARM_CATEGORY_UNSPECIFIED
HarmProbability
HarmProbability.HARM_PROBABILITY_UNSPECIFIED
HarmProbability.HIGH
HarmProbability.LOW
HarmProbability.MEDIUM
HarmProbability.NEGLIGIBLE
HarmSeverity
HarmSeverity.HARM_SEVERITY_HIGH
HarmSeverity.HARM_SEVERITY_LOW
HarmSeverity.HARM_SEVERITY_MEDIUM
HarmSeverity.HARM_SEVERITY_NEGLIGIBLE
HarmSeverity.HARM_SEVERITY_UNSPECIFIED
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 94/152
```
HttpElementLocation
HttpElementLocation.HTTP_IN_BODY
HttpElementLocation.HTTP_IN_COOKIE
HttpElementLocation.HTTP_IN_HEADER
HttpElementLocation.HTTP_IN_PATH
HttpElementLocation.HTTP_IN_QUERY
HttpElementLocation.HTTP_IN_UNSPECIFIED
HttpOptions
HttpOptions.aiohttp_client
HttpOptions.api_version
HttpOptions.async_client_args
HttpOptions.base_url
HttpOptions.base_url_resource_scope
HttpOptions.client_args
HttpOptions.extra_body
HttpOptions.headers
HttpOptions.httpx_async_client
HttpOptions.httpx_client
HttpOptions.retry_options
HttpOptions.timeout
HttpOptionsDict
HttpOptionsDict.api_version
HttpOptionsDict.async_client_args
HttpOptionsDict.base_url
HttpOptionsDict.base_url_resource_scope
HttpOptionsDict.client_args
HttpOptionsDict.extra_body
HttpOptionsDict.headers
HttpOptionsDict.retry_options
HttpOptionsDict.timeout
HttpResponse
HttpResponse.body
HttpResponse.headers
HttpResponseDict
HttpResponseDict.body
HttpResponseDict.headers
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 95/152
```
HttpRetryOptions
HttpRetryOptions.attempts
HttpRetryOptions.exp_base
HttpRetryOptions.http_status_codes
HttpRetryOptions.initial_delay
HttpRetryOptions.jitter
HttpRetryOptions.max_delay
HttpRetryOptionsDict
HttpRetryOptionsDict.attempts
HttpRetryOptionsDict.exp_base
HttpRetryOptionsDict.http_status_codes
HttpRetryOptionsDict.initial_delay
HttpRetryOptionsDict.jitter
HttpRetryOptionsDict.max_delay
Image
Image.gcs_uri
Image.image_bytes
Image.mime_type
```
Image.from_file()
```
```
Image.model_post_init()
```
```
Image.save()
```
```
Image.show()
```
ImageConfig
ImageConfig.aspect_ratio
ImageConfig.image_size
ImageConfig.output_compression_quality
ImageConfig.output_mime_type
ImageConfig.person_generation
ImageConfig.prominent_people
ImageConfigDict
ImageConfigDict.aspect_ratio
ImageConfigDict.image_size
ImageConfigDict.output_compression_quality
ImageConfigDict.output_mime_type
ImageConfigDict.person_generation
ImageConfigDict.prominent_people
ImageDict
ImageDict.gcs_uri
ImageDict.image_bytes
ImageDict.mime_type
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 96/152
```
ImagePromptLanguage
ImagePromptLanguage.auto
ImagePromptLanguage.en
ImagePromptLanguage.es
ImagePromptLanguage.hi
ImagePromptLanguage.ja
ImagePromptLanguage.ko
ImagePromptLanguage.pt
ImagePromptLanguage.zh
ImageSearch
ImageSearchDict
ImportFileConfig
ImportFileConfig.chunking_config
ImportFileConfig.custom_metadata
ImportFileConfig.http_options
ImportFileConfigDict
ImportFileConfigDict.chunking_config
ImportFileConfigDict.custom_metadata
ImportFileConfigDict.http_options
ImportFileOperation
ImportFileOperation.response
```
ImportFileOperation.from_api_response()
```
ImportFileResponse
ImportFileResponse.document_name
ImportFileResponse.parent
ImportFileResponse.sdk_http_response
ImportFileResponseDict
ImportFileResponseDict.document_name
ImportFileResponseDict.parent
ImportFileResponseDict.sdk_http_response
InlinedEmbedContentResponse
InlinedEmbedContentResponse.error
InlinedEmbedContentResponse.response
InlinedEmbedContentResponseDict
InlinedEmbedContentResponseDict.error
InlinedEmbedContentResponseDict.response
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 97/152
```
InlinedRequest
InlinedRequest.config
InlinedRequest.contents
InlinedRequest.metadata
InlinedRequest.model
InlinedRequestDict
InlinedRequestDict.config
InlinedRequestDict.contents
InlinedRequestDict.metadata
InlinedRequestDict.model
InlinedResponse
InlinedResponse.error
InlinedResponse.metadata
InlinedResponse.response
InlinedResponseDict
InlinedResponseDict.error
InlinedResponseDict.metadata
InlinedResponseDict.response
Interval
Interval.end_time
Interval.start_time
IntervalDict
IntervalDict.end_time
IntervalDict.start_time
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 98/152
```
JSONSchema
JSONSchema.additional_properties
JSONSchema.any_of
JSONSchema.default
JSONSchema.defs
JSONSchema.description
JSONSchema.enum
JSONSchema.format
JSONSchema.items
JSONSchema.max_items
JSONSchema.max_length
JSONSchema.max_properties
JSONSchema.maximum
JSONSchema.min_items
JSONSchema.min_length
JSONSchema.min_properties
JSONSchema.minimum
JSONSchema.pattern
JSONSchema.properties
JSONSchema.ref
JSONSchema.required
JSONSchema.title
JSONSchema.type
JSONSchema.unique_items
JSONSchemaType
JSONSchemaType.ARRAY
JSONSchemaType.BOOLEAN
JSONSchemaType.INTEGER
JSONSchemaType.NULL
JSONSchemaType.NUMBER
JSONSchemaType.OBJECT
JSONSchemaType.STRING
JobError
JobError.code
JobError.details
JobError.message
JobErrorDict
JobErrorDict.code
JobErrorDict.details
JobErrorDict.message
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 99/152
```
JobState
JobState.JOB_STATE_CANCELLED
JobState.JOB_STATE_CANCELLING
JobState.JOB_STATE_EXPIRED
JobState.JOB_STATE_FAILED
JobState.JOB_STATE_PARTIALLY_SUCCEEDED
JobState.JOB_STATE_PAUSED
JobState.JOB_STATE_PENDING
JobState.JOB_STATE_QUEUED
JobState.JOB_STATE_RUNNING
JobState.JOB_STATE_SUCCEEDED
JobState.JOB_STATE_UNSPECIFIED
JobState.JOB_STATE_UPDATING
LLMBasedMetricSpec
LLMBasedMetricSpec.additional_config
LLMBasedMetricSpec.judge_autorater_config
LLMBasedMetricSpec.metric_prompt_template
LLMBasedMetricSpec.predefined_rubric_generation_spec
LLMBasedMetricSpec.rubric_generation_spec
LLMBasedMetricSpec.rubric_group_key
LLMBasedMetricSpec.system_instruction
LLMBasedMetricSpecDict
LLMBasedMetricSpecDict.additional_config
LLMBasedMetricSpecDict.judge_autorater_config
LLMBasedMetricSpecDict.metric_prompt_template
LLMBasedMetricSpecDict.predefined_rubric_generation_spec
LLMBasedMetricSpecDict.rubric_generation_spec
LLMBasedMetricSpecDict.rubric_group_key
LLMBasedMetricSpecDict.system_instruction
Language
Language.LANGUAGE_UNSPECIFIED
Language.PYTHON
LatLng
LatLng.latitude
LatLng.longitude
LatLngDict
LatLngDict.latitude
LatLngDict.longitudeSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 100/152
```
ListBatchJobsConfig
ListBatchJobsConfig.filter
ListBatchJobsConfig.http_options
ListBatchJobsConfig.page_size
ListBatchJobsConfig.page_token
ListBatchJobsConfigDict
ListBatchJobsConfigDict.filter
ListBatchJobsConfigDict.http_options
ListBatchJobsConfigDict.page_size
ListBatchJobsConfigDict.page_token
ListBatchJobsResponse
ListBatchJobsResponse.batch_jobs
ListBatchJobsResponse.next_page_token
ListBatchJobsResponse.sdk_http_response
ListBatchJobsResponseDict
ListBatchJobsResponseDict.batch_jobs
ListBatchJobsResponseDict.next_page_token
ListBatchJobsResponseDict.sdk_http_response
ListCachedContentsConfig
ListCachedContentsConfig.http_options
ListCachedContentsConfig.page_size
ListCachedContentsConfig.page_token
ListCachedContentsConfigDict
ListCachedContentsConfigDict.http_options
ListCachedContentsConfigDict.page_size
ListCachedContentsConfigDict.page_token
ListCachedContentsResponse
ListCachedContentsResponse.cached_contents
ListCachedContentsResponse.next_page_token
ListCachedContentsResponse.sdk_http_response
ListCachedContentsResponseDict
ListCachedContentsResponseDict.cached_contents
ListCachedContentsResponseDict.next_page_token
ListCachedContentsResponseDict.sdk_http_response
ListDocumentsConfig
ListDocumentsConfig.http_options
ListDocumentsConfig.page_size
ListDocumentsConfig.page_tokenSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 101/152
```
ListDocumentsConfigDict
ListDocumentsConfigDict.http_options
ListDocumentsConfigDict.page_size
ListDocumentsConfigDict.page_token
ListDocumentsResponse
ListDocumentsResponse.documents
ListDocumentsResponse.next_page_token
ListDocumentsResponse.sdk_http_response
ListDocumentsResponseDict
ListDocumentsResponseDict.documents
ListDocumentsResponseDict.next_page_token
ListDocumentsResponseDict.sdk_http_response
ListFileSearchStoresConfig
ListFileSearchStoresConfig.http_options
ListFileSearchStoresConfig.page_size
ListFileSearchStoresConfig.page_token
ListFileSearchStoresConfigDict
ListFileSearchStoresConfigDict.http_options
ListFileSearchStoresConfigDict.page_size
ListFileSearchStoresConfigDict.page_token
ListFileSearchStoresResponse
ListFileSearchStoresResponse.file_search_stores
ListFileSearchStoresResponse.next_page_token
ListFileSearchStoresResponse.sdk_http_response
ListFileSearchStoresResponseDict
ListFileSearchStoresResponseDict.file_search_stores
ListFileSearchStoresResponseDict.next_page_token
ListFileSearchStoresResponseDict.sdk_http_response
ListFilesConfig
ListFilesConfig.http_options
ListFilesConfig.page_size
ListFilesConfig.page_token
ListFilesConfigDict
ListFilesConfigDict.http_options
ListFilesConfigDict.page_size
ListFilesConfigDict.page_token
ListFilesResponse
ListFilesResponse.files
ListFilesResponse.next_page_token
ListFilesResponse.sdk_http_response
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 102/152
```
ListFilesResponseDict
ListFilesResponseDict.files
ListFilesResponseDict.next_page_token
ListFilesResponseDict.sdk_http_response
ListModelsConfig
ListModelsConfig.filter
ListModelsConfig.http_options
ListModelsConfig.page_size
ListModelsConfig.page_token
ListModelsConfig.query_base
ListModelsConfigDict
ListModelsConfigDict.filter
ListModelsConfigDict.http_options
ListModelsConfigDict.page_size
ListModelsConfigDict.page_token
ListModelsConfigDict.query_base
ListModelsResponse
ListModelsResponse.models
ListModelsResponse.next_page_token
ListModelsResponse.sdk_http_response
ListModelsResponseDict
ListModelsResponseDict.models
ListModelsResponseDict.next_page_token
ListModelsResponseDict.sdk_http_response
ListTuningJobsConfig
ListTuningJobsConfig.filter
ListTuningJobsConfig.http_options
ListTuningJobsConfig.page_size
ListTuningJobsConfig.page_token
ListTuningJobsConfigDict
ListTuningJobsConfigDict.filter
ListTuningJobsConfigDict.http_options
ListTuningJobsConfigDict.page_size
ListTuningJobsConfigDict.page_token
ListTuningJobsResponse
ListTuningJobsResponse.next_page_token
ListTuningJobsResponse.sdk_http_response
ListTuningJobsResponse.tuning_jobsSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 103/152
```
ListTuningJobsResponseDict
ListTuningJobsResponseDict.next_page_token
ListTuningJobsResponseDict.sdk_http_response
ListTuningJobsResponseDict.tuning_jobs
LiveClientContent
LiveClientContent.turn_complete
LiveClientContent.turns
LiveClientContentDict
LiveClientContentDict.turn_complete
LiveClientContentDict.turns
LiveClientMessage
LiveClientMessage.client_content
LiveClientMessage.realtime_input
LiveClientMessage.setup
LiveClientMessage.tool_response
LiveClientMessageDict
LiveClientMessageDict.client_content
LiveClientMessageDict.realtime_input
LiveClientMessageDict.setup
LiveClientMessageDict.tool_response
LiveClientRealtimeInput
LiveClientRealtimeInput.activity_end
LiveClientRealtimeInput.activity_start
LiveClientRealtimeInput.audio
LiveClientRealtimeInput.audio_stream_end
LiveClientRealtimeInput.media_chunks
LiveClientRealtimeInput.text
LiveClientRealtimeInput.video
LiveClientRealtimeInputDict
LiveClientRealtimeInputDict.activity_end
LiveClientRealtimeInputDict.activity_start
LiveClientRealtimeInputDict.audio
LiveClientRealtimeInputDict.audio_stream_end
LiveClientRealtimeInputDict.media_chunks
LiveClientRealtimeInputDict.text
LiveClientRealtimeInputDict.video
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 104/152
```
LiveClientSetup
LiveClientSetup.context_window_compression
LiveClientSetup.explicit_vad_signal
LiveClientSetup.generation_config
LiveClientSetup.input_audio_transcription
LiveClientSetup.model
LiveClientSetup.output_audio_transcription
LiveClientSetup.proactivity
LiveClientSetup.session_resumption
LiveClientSetup.system_instruction
LiveClientSetup.tools
LiveClientSetupDict
LiveClientSetupDict.context_window_compression
LiveClientSetupDict.explicit_vad_signal
LiveClientSetupDict.generation_config
LiveClientSetupDict.input_audio_transcription
LiveClientSetupDict.model
LiveClientSetupDict.output_audio_transcription
LiveClientSetupDict.proactivity
LiveClientSetupDict.session_resumption
LiveClientSetupDict.system_instruction
LiveClientSetupDict.tools
LiveClientToolResponse
LiveClientToolResponse.function_responses
LiveClientToolResponseDict
LiveClientToolResponseDict.function_responses
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 105/152
```
LiveConnectConfig
LiveConnectConfig.context_window_compression
LiveConnectConfig.enable_affective_dialog
LiveConnectConfig.explicit_vad_signal
LiveConnectConfig.generation_config
LiveConnectConfig.http_options
LiveConnectConfig.input_audio_transcription
LiveConnectConfig.max_output_tokens
LiveConnectConfig.media_resolution
LiveConnectConfig.output_audio_transcription
LiveConnectConfig.proactivity
LiveConnectConfig.realtime_input_config
LiveConnectConfig.response_modalities
LiveConnectConfig.seed
LiveConnectConfig.session_resumption
LiveConnectConfig.speech_config
LiveConnectConfig.system_instruction
LiveConnectConfig.temperature
LiveConnectConfig.thinking_config
LiveConnectConfig.tools
LiveConnectConfig.top_k
LiveConnectConfig.top_p
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 106/152
```
LiveConnectConfigDict
LiveConnectConfigDict.context_window_compression
LiveConnectConfigDict.enable_affective_dialog
LiveConnectConfigDict.explicit_vad_signal
LiveConnectConfigDict.generation_config
LiveConnectConfigDict.http_options
LiveConnectConfigDict.input_audio_transcription
LiveConnectConfigDict.max_output_tokens
LiveConnectConfigDict.media_resolution
LiveConnectConfigDict.output_audio_transcription
LiveConnectConfigDict.proactivity
LiveConnectConfigDict.realtime_input_config
LiveConnectConfigDict.response_modalities
LiveConnectConfigDict.seed
LiveConnectConfigDict.session_resumption
LiveConnectConfigDict.speech_config
LiveConnectConfigDict.system_instruction
LiveConnectConfigDict.temperature
LiveConnectConfigDict.thinking_config
LiveConnectConfigDict.tools
LiveConnectConfigDict.top_k
LiveConnectConfigDict.top_p
LiveConnectConstraints
LiveConnectConstraints.config
LiveConnectConstraints.model
LiveConnectConstraintsDict
LiveConnectConstraintsDict.config
LiveConnectConstraintsDict.model
LiveConnectParameters
LiveConnectParameters.config
LiveConnectParameters.model
LiveConnectParametersDict
LiveConnectParametersDict.config
LiveConnectParametersDict.model
LiveMusicClientContent
LiveMusicClientContent.weighted_prompts
LiveMusicClientContentDict
LiveMusicClientContentDict.weighted_promptsSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 107/152
```
LiveMusicClientMessage
LiveMusicClientMessage.client_content
LiveMusicClientMessage.music_generation_config
LiveMusicClientMessage.playback_control
LiveMusicClientMessage.setup
LiveMusicClientMessageDict
LiveMusicClientMessageDict.client_content
LiveMusicClientMessageDict.music_generation_config
LiveMusicClientMessageDict.playback_control
LiveMusicClientMessageDict.setup
LiveMusicClientSetup
LiveMusicClientSetup.model
LiveMusicClientSetupDict
LiveMusicClientSetupDict.model
LiveMusicConnectParameters
LiveMusicConnectParameters.model
LiveMusicConnectParametersDict
LiveMusicConnectParametersDict.model
LiveMusicFilteredPrompt
LiveMusicFilteredPrompt.filtered_reason
LiveMusicFilteredPrompt.text
LiveMusicFilteredPromptDict
LiveMusicFilteredPromptDict.filtered_reason
LiveMusicFilteredPromptDict.text
LiveMusicGenerationConfig
LiveMusicGenerationConfig.bpm
LiveMusicGenerationConfig.brightness
LiveMusicGenerationConfig.density
LiveMusicGenerationConfig.guidance
LiveMusicGenerationConfig.music_generation_mode
LiveMusicGenerationConfig.mute_bass
LiveMusicGenerationConfig.mute_drums
LiveMusicGenerationConfig.only_bass_and_drums
LiveMusicGenerationConfig.scale
LiveMusicGenerationConfig.seed
LiveMusicGenerationConfig.temperature
LiveMusicGenerationConfig.top_k
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 108/152
```
LiveMusicGenerationConfigDict
LiveMusicGenerationConfigDict.bpm
LiveMusicGenerationConfigDict.brightness
LiveMusicGenerationConfigDict.density
LiveMusicGenerationConfigDict.guidance
LiveMusicGenerationConfigDict.music_generation_mode
LiveMusicGenerationConfigDict.mute_bass
LiveMusicGenerationConfigDict.mute_drums
LiveMusicGenerationConfigDict.only_bass_and_drums
LiveMusicGenerationConfigDict.scale
LiveMusicGenerationConfigDict.seed
LiveMusicGenerationConfigDict.temperature
LiveMusicGenerationConfigDict.top_k
LiveMusicPlaybackControl
LiveMusicPlaybackControl.PAUSE
LiveMusicPlaybackControl.PLAY
LiveMusicPlaybackControl.PLAYBACK_CONTROL_UNSPECIFIED
LiveMusicPlaybackControl.RESET_CONTEXT
LiveMusicPlaybackControl.STOP
LiveMusicServerContent
LiveMusicServerContent.audio_chunks
LiveMusicServerContentDict
LiveMusicServerContentDict.audio_chunks
LiveMusicServerMessage
LiveMusicServerMessage.filtered_prompt
LiveMusicServerMessage.server_content
LiveMusicServerMessage.setup_complete
LiveMusicServerMessageDict
LiveMusicServerMessageDict.filtered_prompt
LiveMusicServerMessageDict.server_content
LiveMusicServerMessageDict.setup_complete
LiveMusicServerSetupComplete
LiveMusicServerSetupCompleteDict
LiveMusicSetConfigParameters
LiveMusicSetConfigParameters.music_generation_config
LiveMusicSetConfigParametersDict
LiveMusicSetConfigParametersDict.music_generation_config
LiveMusicSetWeightedPromptsParameters
LiveMusicSetWeightedPromptsParameters.weighted_prompts
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 109/152
```
LiveMusicSetWeightedPromptsParametersDict
LiveMusicSetWeightedPromptsParametersDict.weighted_prompts
LiveMusicSourceMetadata
LiveMusicSourceMetadata.client_content
LiveMusicSourceMetadata.music_generation_config
LiveMusicSourceMetadataDict
LiveMusicSourceMetadataDict.client_content
LiveMusicSourceMetadataDict.music_generation_config
LiveSendRealtimeInputParameters
LiveSendRealtimeInputParameters.activity_end
LiveSendRealtimeInputParameters.activity_start
LiveSendRealtimeInputParameters.audio
LiveSendRealtimeInputParameters.audio_stream_end
LiveSendRealtimeInputParameters.media
LiveSendRealtimeInputParameters.text
LiveSendRealtimeInputParameters.video
LiveSendRealtimeInputParametersDict
LiveSendRealtimeInputParametersDict.activity_end
LiveSendRealtimeInputParametersDict.activity_start
LiveSendRealtimeInputParametersDict.audio
LiveSendRealtimeInputParametersDict.audio_stream_end
LiveSendRealtimeInputParametersDict.media
LiveSendRealtimeInputParametersDict.text
LiveSendRealtimeInputParametersDict.video
LiveServerContent
LiveServerContent.generation_complete
LiveServerContent.grounding_metadata
LiveServerContent.input_transcription
LiveServerContent.interrupted
LiveServerContent.model_turn
LiveServerContent.output_transcription
LiveServerContent.turn_complete
LiveServerContent.turn_complete_reason
LiveServerContent.url_context_metadata
LiveServerContent.waiting_for_input
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 110/152
```
LiveServerContentDict
LiveServerContentDict.generation_complete
LiveServerContentDict.grounding_metadata
LiveServerContentDict.input_transcription
LiveServerContentDict.interrupted
LiveServerContentDict.model_turn
LiveServerContentDict.output_transcription
LiveServerContentDict.turn_complete
LiveServerContentDict.turn_complete_reason
LiveServerContentDict.url_context_metadata
LiveServerContentDict.waiting_for_input
LiveServerGoAway
LiveServerGoAway.time_left
LiveServerGoAwayDict
LiveServerGoAwayDict.time_left
LiveServerMessage
LiveServerMessage.go_away
LiveServerMessage.server_content
LiveServerMessage.session_resumption_update
LiveServerMessage.setup_complete
LiveServerMessage.tool_call
LiveServerMessage.tool_call_cancellation
LiveServerMessage.usage_metadata
LiveServerMessage.voice_activity
LiveServerMessage.voice_activity_detection_signal
LiveServerMessage.data
LiveServerMessage.text
LiveServerMessageDict
LiveServerMessageDict.go_away
LiveServerMessageDict.server_content
LiveServerMessageDict.session_resumption_update
LiveServerMessageDict.setup_complete
LiveServerMessageDict.tool_call
LiveServerMessageDict.tool_call_cancellation
LiveServerMessageDict.usage_metadata
LiveServerMessageDict.voice_activity
LiveServerMessageDict.voice_activity_detection_signal
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 111/152
```
LiveServerSessionResumptionUpdate
LiveServerSessionResumptionUpdate.last_consumed_client_message_index
LiveServerSessionResumptionUpdate.new_handle
LiveServerSessionResumptionUpdate.resumable
LiveServerSessionResumptionUpdateDict
LiveServerSessionResumptionUpdateDict.last_consumed_client_message_index
LiveServerSessionResumptionUpdateDict.new_handle
LiveServerSessionResumptionUpdateDict.resumable
LiveServerSetupComplete
LiveServerSetupComplete.session_id
LiveServerSetupCompleteDict
LiveServerSetupCompleteDict.session_id
LiveServerToolCall
LiveServerToolCall.function_calls
LiveServerToolCallCancellation
LiveServerToolCallCancellation.ids
LiveServerToolCallCancellationDict
LiveServerToolCallCancellationDict.ids
LiveServerToolCallDict
LiveServerToolCallDict.function_calls
LogprobsResult
LogprobsResult.chosen_candidates
LogprobsResult.top_candidates
LogprobsResultCandidate
LogprobsResultCandidate.log_probability
LogprobsResultCandidate.token
LogprobsResultCandidate.token_id
LogprobsResultCandidateDict
LogprobsResultCandidateDict.log_probability
LogprobsResultCandidateDict.token
LogprobsResultCandidateDict.token_id
LogprobsResultDict
LogprobsResultDict.chosen_candidates
LogprobsResultDict.top_candidates
LogprobsResultTopCandidates
LogprobsResultTopCandidates.candidates
LogprobsResultTopCandidatesDict
LogprobsResultTopCandidatesDict.candidatesSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 112/152
```
MaskReferenceConfig
MaskReferenceConfig.mask_dilation
MaskReferenceConfig.mask_mode
MaskReferenceConfig.segmentation_classes
MaskReferenceConfigDict
MaskReferenceConfigDict.mask_dilation
MaskReferenceConfigDict.mask_mode
MaskReferenceConfigDict.segmentation_classes
MaskReferenceImage
MaskReferenceImage.config
MaskReferenceImage.mask_image_config
MaskReferenceImage.reference_id
MaskReferenceImage.reference_image
MaskReferenceImage.reference_type
MaskReferenceImageDict
MaskReferenceImageDict.config
MaskReferenceImageDict.reference_id
MaskReferenceImageDict.reference_image
MaskReferenceImageDict.reference_type
MaskReferenceMode
MaskReferenceMode.MASK_MODE_BACKGROUND
MaskReferenceMode.MASK_MODE_DEFAULT
MaskReferenceMode.MASK_MODE_FOREGROUND
MaskReferenceMode.MASK_MODE_SEMANTIC
MaskReferenceMode.MASK_MODE_USER_PROVIDED
McpServer
McpServer.name
McpServer.streamable_http_transport
McpServerDict
McpServerDict.name
McpServerDict.streamable_http_transport
MediaModality
MediaModality.AUDIO
MediaModality.DOCUMENT
MediaModality.IMAGE
MediaModality.MODALITY_UNSPECIFIED
MediaModality.TEXT
MediaModality.VIDEOSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 113/152
```
MediaResolution
MediaResolution.MEDIA_RESOLUTION_HIGH
MediaResolution.MEDIA_RESOLUTION_LOW
MediaResolution.MEDIA_RESOLUTION_MEDIUM
MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED
Metric
Metric.aggregate_summary_fn
Metric.custom_function
Metric.judge_model_system_instruction
Metric.name
Metric.parse_and_reduce_fn
Metric.prompt_template
Metric.return_raw_output
```
Metric.model_post_init()
```
```
Metric.to_yaml_file()
```
Metric.validate_name
MetricDict
MetricDict.aggregate_summary_fn
MetricDict.custom_function
MetricDict.judge_model_system_instruction
MetricDict.name
MetricDict.parse_and_reduce_fn
MetricDict.prompt_template
MetricDict.return_raw_output
Modality
Modality.AUDIO
Modality.IMAGE
Modality.MODALITY_UNSPECIFIED
Modality.TEXT
ModalityTokenCount
ModalityTokenCount.modality
ModalityTokenCount.token_count
ModalityTokenCountDict
ModalityTokenCountDict.modality
ModalityTokenCountDict.token_count
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 114/152
```
Model
Model.checkpoints
Model.default_checkpoint_id
Model.description
Model.display_name
Model.endpoints
Model.input_token_limit
Model.labels
Model.max_temperature
Model.name
Model.output_token_limit
Model.supported_actions
Model.temperature
Model.thinking
Model.top_k
Model.top_p
Model.tuned_model_info
Model.version
ModelArmorConfig
ModelArmorConfig.prompt_template_name
ModelArmorConfig.response_template_name
ModelArmorConfigDict
ModelArmorConfigDict.prompt_template_name
ModelArmorConfigDict.response_template_name
ModelContent
ModelContent.parts
ModelContent.role
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 115/152
```
ModelDict
ModelDict.checkpoints
ModelDict.default_checkpoint_id
ModelDict.description
ModelDict.display_name
ModelDict.endpoints
ModelDict.input_token_limit
ModelDict.labels
ModelDict.max_temperature
ModelDict.name
ModelDict.output_token_limit
ModelDict.supported_actions
ModelDict.temperature
ModelDict.thinking
ModelDict.top_k
ModelDict.top_p
ModelDict.tuned_model_info
ModelDict.version
ModelSelectionConfig
ModelSelectionConfig.feature_selection_preference
ModelSelectionConfigDict
ModelSelectionConfigDict.feature_selection_preference
MultiSpeakerVoiceConfig
MultiSpeakerVoiceConfig.speaker_voice_configs
MultiSpeakerVoiceConfigDict
MultiSpeakerVoiceConfigDict.speaker_voice_configs
MusicGenerationMode
MusicGenerationMode.DIVERSITY
MusicGenerationMode.MUSIC_GENERATION_MODE_UNSPECIFIED
MusicGenerationMode.QUALITY
MusicGenerationMode.VOCALIZATION
Operation
Operation.done
Operation.error
```
Operation.from_api_response()
```
Operation.metadata
Operation.name
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 116/152
```
Outcome
Outcome.OUTCOME_DEADLINE_EXCEEDED
Outcome.OUTCOME_FAILED
Outcome.OUTCOME_OK
Outcome.OUTCOME_UNSPECIFIED
OutputConfig
OutputConfig.gcs_destination
OutputConfigDict
OutputConfigDict.gcs_destination
PairwiseMetricSpec
PairwiseMetricSpec.baseline_response_field_name
PairwiseMetricSpec.candidate_response_field_name
PairwiseMetricSpec.custom_output_format_config
PairwiseMetricSpec.metric_prompt_template
PairwiseMetricSpec.system_instruction
PairwiseMetricSpecDict
PairwiseMetricSpecDict.baseline_response_field_name
PairwiseMetricSpecDict.candidate_response_field_name
PairwiseMetricSpecDict.custom_output_format_config
PairwiseMetricSpecDict.metric_prompt_template
PairwiseMetricSpecDict.system_instruction
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 117/152
```
Part
Part.code_execution_result
Part.executable_code
Part.file_data
Part.function_call
Part.function_response
Part.inline_data
Part.media_resolution
Part.text
Part.thought
Part.thought_signature
Part.video_metadata
```
Part.from_bytes()
```
```
Part.from_code_execution_result()
```
```
Part.from_executable_code()
```
```
Part.from_function_call()
```
```
Part.from_function_response()
```
```
Part.from_text()
```
```
Part.from_uri()
```
```
Part.as_image()
```
PartDict
PartDict.code_execution_result
PartDict.executable_code
PartDict.file_data
PartDict.function_call
PartDict.function_response
PartDict.inline_data
PartDict.media_resolution
PartDict.text
PartDict.thought
PartDict.thought_signature
PartDict.video_metadata
PartMediaResolution
PartMediaResolution.level
PartMediaResolution.num_tokens
PartMediaResolutionDict
PartMediaResolutionDict.level
PartMediaResolutionDict.num_tokensSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 118/152
```
PartMediaResolutionLevel
PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH
PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM
PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH
PartMediaResolutionLevel.MEDIA_RESOLUTION_UNSPECIFIED
PartialArg
PartialArg.bool_value
PartialArg.json_path
PartialArg.null_value
PartialArg.number_value
PartialArg.string_value
PartialArg.will_continue
PartialArgDict
PartialArgDict.bool_value
PartialArgDict.json_path
PartialArgDict.null_value
PartialArgDict.number_value
PartialArgDict.string_value
PartialArgDict.will_continue
PartnerModelTuningSpec
PartnerModelTuningSpec.hyper_parameters
PartnerModelTuningSpec.training_dataset_uri
PartnerModelTuningSpec.validation_dataset_uri
PartnerModelTuningSpecDict
PartnerModelTuningSpecDict.hyper_parameters
PartnerModelTuningSpecDict.training_dataset_uri
PartnerModelTuningSpecDict.validation_dataset_uri
PersonGeneration
PersonGeneration.ALLOW_ADULT
PersonGeneration.ALLOW_ALL
PersonGeneration.DONT_ALLOW
PhishBlockThreshold
PhishBlockThreshold.BLOCK_HIGHER_AND_ABOVE
PhishBlockThreshold.BLOCK_HIGH_AND_ABOVE
PhishBlockThreshold.BLOCK_LOW_AND_ABOVE
PhishBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
PhishBlockThreshold.BLOCK_ONLY_EXTREMELY_HIGH
PhishBlockThreshold.BLOCK_VERY_HIGH_AND_ABOVE
PhishBlockThreshold.PHISH_BLOCK_THRESHOLD_UNSPECIFIED
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 119/152
```
PointwiseMetricSpec
PointwiseMetricSpec.custom_output_format_config
PointwiseMetricSpec.metric_prompt_template
PointwiseMetricSpec.system_instruction
PointwiseMetricSpecDict
PointwiseMetricSpecDict.custom_output_format_config
PointwiseMetricSpecDict.metric_prompt_template
PointwiseMetricSpecDict.system_instruction
PreTunedModel
PreTunedModel.base_model
PreTunedModel.checkpoint_id
PreTunedModel.tuned_model_name
PreTunedModelDict
PreTunedModelDict.base_model
PreTunedModelDict.checkpoint_id
PreTunedModelDict.tuned_model_name
PrebuiltVoiceConfig
PrebuiltVoiceConfig.voice_name
PrebuiltVoiceConfigDict
PrebuiltVoiceConfigDict.voice_name
PredefinedMetricSpec
PredefinedMetricSpec.metric_spec_name
PredefinedMetricSpec.metric_spec_parameters
PredefinedMetricSpecDict
PredefinedMetricSpecDict.metric_spec_name
PredefinedMetricSpecDict.metric_spec_parameters
PreferenceOptimizationDataStats
PreferenceOptimizationDataStats.score_variance_per_example_distribution
PreferenceOptimizationDataStats.scores_distribution
PreferenceOptimizationDataStats.total_billable_token_count
PreferenceOptimizationDataStats.tuning_dataset_example_count
PreferenceOptimizationDataStats.tuning_step_count
PreferenceOptimizationDataStats.user_dataset_examples
PreferenceOptimizationDataStats.user_input_token_distribution
PreferenceOptimizationDataStats.user_output_token_distribution
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 120/152
```
PreferenceOptimizationDataStatsDict
PreferenceOptimizationDataStatsDict.score_variance_per_example_distribution
PreferenceOptimizationDataStatsDict.scores_distribution
PreferenceOptimizationDataStatsDict.total_billable_token_count
PreferenceOptimizationDataStatsDict.tuning_dataset_example_count
PreferenceOptimizationDataStatsDict.tuning_step_count
PreferenceOptimizationDataStatsDict.user_dataset_examples
PreferenceOptimizationDataStatsDict.user_input_token_distribution
PreferenceOptimizationDataStatsDict.user_output_token_distribution
PreferenceOptimizationHyperParameters
PreferenceOptimizationHyperParameters.adapter_size
PreferenceOptimizationHyperParameters.beta
PreferenceOptimizationHyperParameters.epoch_count
PreferenceOptimizationHyperParameters.learning_rate_multiplier
PreferenceOptimizationHyperParametersDict
PreferenceOptimizationHyperParametersDict.adapter_size
PreferenceOptimizationHyperParametersDict.beta
PreferenceOptimizationHyperParametersDict.epoch_count
PreferenceOptimizationHyperParametersDict.learning_rate_multiplier
PreferenceOptimizationSpec
PreferenceOptimizationSpec.export_last_checkpoint_only
PreferenceOptimizationSpec.hyper_parameters
PreferenceOptimizationSpec.training_dataset_uri
PreferenceOptimizationSpec.validation_dataset_uri
PreferenceOptimizationSpecDict
PreferenceOptimizationSpecDict.export_last_checkpoint_only
PreferenceOptimizationSpecDict.hyper_parameters
PreferenceOptimizationSpecDict.training_dataset_uri
PreferenceOptimizationSpecDict.validation_dataset_uri
ProactivityConfig
ProactivityConfig.proactive_audio
ProactivityConfigDict
ProactivityConfigDict.proactive_audio
ProductImage
ProductImage.product_image
ProductImageDict
ProductImageDict.product_image
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 121/152
```
ProjectOperation
ProjectOperation.done
ProjectOperation.error
ProjectOperation.metadata
ProjectOperation.name
ProjectOperationDict
ProjectOperationDict.done
ProjectOperationDict.error
ProjectOperationDict.metadata
ProjectOperationDict.name
ProminentPeople
ProminentPeople.ALLOW_PROMINENT_PEOPLE
ProminentPeople.BLOCK_PROMINENT_PEOPLE
ProminentPeople.PROMINENT_PEOPLE_UNSPECIFIED
RagChunk
RagChunk.page_span
RagChunk.text
RagChunkDict
RagChunkDict.page_span
RagChunkDict.text
RagChunkPageSpan
RagChunkPageSpan.first_page
RagChunkPageSpan.last_page
RagChunkPageSpanDict
RagChunkPageSpanDict.first_page
RagChunkPageSpanDict.last_page
RagRetrievalConfig
RagRetrievalConfig.filter
RagRetrievalConfig.hybrid_search
RagRetrievalConfig.ranking
RagRetrievalConfig.top_k
RagRetrievalConfigDict
RagRetrievalConfigDict.filter
RagRetrievalConfigDict.hybrid_search
RagRetrievalConfigDict.ranking
RagRetrievalConfigDict.top_k
RagRetrievalConfigFilter
RagRetrievalConfigFilter.metadata_filter
RagRetrievalConfigFilter.vector_distance_threshold
RagRetrievalConfigFilter.vector_similarity_threshold
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 122/152
```
RagRetrievalConfigFilterDict
RagRetrievalConfigFilterDict.metadata_filter
RagRetrievalConfigFilterDict.vector_distance_threshold
RagRetrievalConfigFilterDict.vector_similarity_threshold
RagRetrievalConfigHybridSearch
RagRetrievalConfigHybridSearch.alpha
RagRetrievalConfigHybridSearchDict
RagRetrievalConfigHybridSearchDict.alpha
RagRetrievalConfigRanking
RagRetrievalConfigRanking.llm_ranker
RagRetrievalConfigRanking.rank_service
RagRetrievalConfigRankingDict
RagRetrievalConfigRankingDict.llm_ranker
RagRetrievalConfigRankingDict.rank_service
RagRetrievalConfigRankingLlmRanker
RagRetrievalConfigRankingLlmRanker.model_name
RagRetrievalConfigRankingLlmRankerDict
RagRetrievalConfigRankingLlmRankerDict.model_name
RagRetrievalConfigRankingRankService
RagRetrievalConfigRankingRankService.model_name
RagRetrievalConfigRankingRankServiceDict
RagRetrievalConfigRankingRankServiceDict.model_name
RawReferenceImage
RawReferenceImage.reference_id
RawReferenceImage.reference_image
RawReferenceImage.reference_type
RawReferenceImageDict
RawReferenceImageDict.reference_id
RawReferenceImageDict.reference_image
RawReferenceImageDict.reference_type
RealtimeInputConfig
RealtimeInputConfig.activity_handling
RealtimeInputConfig.automatic_activity_detection
RealtimeInputConfig.turn_coverage
RealtimeInputConfigDict
RealtimeInputConfigDict.activity_handling
RealtimeInputConfigDict.automatic_activity_detection
RealtimeInputConfigDict.turn_coverageSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 123/152
```
RecontextImageConfig
RecontextImageConfig.add_watermark
RecontextImageConfig.base_steps
RecontextImageConfig.enhance_prompt
RecontextImageConfig.http_options
RecontextImageConfig.labels
RecontextImageConfig.number_of_images
RecontextImageConfig.output_compression_quality
RecontextImageConfig.output_gcs_uri
RecontextImageConfig.output_mime_type
RecontextImageConfig.person_generation
RecontextImageConfig.safety_filter_level
RecontextImageConfig.seed
RecontextImageConfigDict
RecontextImageConfigDict.add_watermark
RecontextImageConfigDict.base_steps
RecontextImageConfigDict.enhance_prompt
RecontextImageConfigDict.http_options
RecontextImageConfigDict.labels
RecontextImageConfigDict.number_of_images
RecontextImageConfigDict.output_compression_quality
RecontextImageConfigDict.output_gcs_uri
RecontextImageConfigDict.output_mime_type
RecontextImageConfigDict.person_generation
RecontextImageConfigDict.safety_filter_level
RecontextImageConfigDict.seed
RecontextImageResponse
RecontextImageResponse.generated_images
RecontextImageResponseDict
RecontextImageResponseDict.generated_images
RecontextImageSource
RecontextImageSource.person_image
RecontextImageSource.product_images
RecontextImageSource.prompt
RecontextImageSourceDict
RecontextImageSourceDict.person_image
RecontextImageSourceDict.product_images
RecontextImageSourceDict.promptSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 124/152
```
RegisterFilesConfig
RegisterFilesConfig.http_options
RegisterFilesConfig.should_return_http_response
RegisterFilesConfigDict
RegisterFilesConfigDict.http_options
RegisterFilesConfigDict.should_return_http_response
RegisterFilesResponse
RegisterFilesResponse.files
RegisterFilesResponse.sdk_http_response
RegisterFilesResponseDict
RegisterFilesResponseDict.files
RegisterFilesResponseDict.sdk_http_response
ReplayFile
ReplayFile.interactions
ReplayFile.replay_id
ReplayFileDict
ReplayFileDict.interactions
ReplayFileDict.replay_id
ReplayInteraction
ReplayInteraction.request
ReplayInteraction.response
ReplayInteractionDict
ReplayInteractionDict.request
ReplayInteractionDict.response
ReplayRequest
ReplayRequest.body_segments
ReplayRequest.headers
ReplayRequest.method
ReplayRequest.url
ReplayRequestDict
ReplayRequestDict.body_segments
ReplayRequestDict.headers
ReplayRequestDict.method
ReplayRequestDict.url
ReplayResponse
ReplayResponse.body_segments
ReplayResponse.headers
ReplayResponse.sdk_response_segments
ReplayResponse.status_code
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 125/152
```
ReplayResponseDict
ReplayResponseDict.body_segments
ReplayResponseDict.headers
ReplayResponseDict.sdk_response_segments
ReplayResponseDict.status_code
ReplicatedVoiceConfig
ReplicatedVoiceConfig.mime_type
ReplicatedVoiceConfig.voice_sample_audio
ReplicatedVoiceConfigDict
ReplicatedVoiceConfigDict.mime_type
ReplicatedVoiceConfigDict.voice_sample_audio
ResourceScope
ResourceScope.COLLECTION
Retrieval
Retrieval.disable_attribution
Retrieval.external_api
Retrieval.vertex_ai_search
Retrieval.vertex_rag_store
RetrievalConfig
RetrievalConfig.language_code
RetrievalConfig.lat_lng
RetrievalConfigDict
RetrievalConfigDict.language_code
RetrievalConfigDict.lat_lng
RetrievalDict
RetrievalDict.disable_attribution
RetrievalDict.external_api
RetrievalDict.vertex_ai_search
RetrievalDict.vertex_rag_store
RetrievalMetadata
RetrievalMetadata.google_search_dynamic_retrieval_score
RetrievalMetadataDict
RetrievalMetadataDict.google_search_dynamic_retrieval_score
RougeSpec
RougeSpec.rouge_type
RougeSpec.split_summaries
RougeSpec.use_stemmer
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 126/152
```
RougeSpecDict
RougeSpecDict.rouge_type
RougeSpecDict.split_summaries
RougeSpecDict.use_stemmer
RubricContentType
RubricContentType.NL_QUESTION_ANSWER
RubricContentType.PROPERTY
RubricContentType.PYTHON_CODE_ASSERTION
RubricContentType.RUBRIC_CONTENT_TYPE_UNSPECIFIED
RubricGenerationSpec
RubricGenerationSpec.prompt_template
RubricGenerationSpec.rubric_content_type
RubricGenerationSpec.rubric_type_ontology
RubricGenerationSpecDict
RubricGenerationSpecDict.prompt_template
RubricGenerationSpecDict.rubric_content_type
RubricGenerationSpecDict.rubric_type_ontology
SafetyAttributes
SafetyAttributes.categories
SafetyAttributes.content_type
SafetyAttributes.scores
SafetyAttributesDict
SafetyAttributesDict.categories
SafetyAttributesDict.content_type
SafetyAttributesDict.scores
SafetyFilterLevel
SafetyFilterLevel.BLOCK_LOW_AND_ABOVE
SafetyFilterLevel.BLOCK_MEDIUM_AND_ABOVE
SafetyFilterLevel.BLOCK_NONE
SafetyFilterLevel.BLOCK_ONLY_HIGH
SafetyRating
SafetyRating.blocked
SafetyRating.category
SafetyRating.overwritten_threshold
SafetyRating.probability
SafetyRating.probability_score
SafetyRating.severity
SafetyRating.severity_scoreSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 127/152
```
SafetyRatingDict
SafetyRatingDict.blocked
SafetyRatingDict.category
SafetyRatingDict.overwritten_threshold
SafetyRatingDict.probability
SafetyRatingDict.probability_score
SafetyRatingDict.severity
SafetyRatingDict.severity_score
SafetySetting
SafetySetting.category
SafetySetting.method
SafetySetting.threshold
SafetySettingDict
SafetySettingDict.category
SafetySettingDict.method
SafetySettingDict.threshold
Scale
Scale.A_FLAT_MAJOR_F_MINOR
Scale.A_MAJOR_G_FLAT_MINOR
Scale.B_FLAT_MAJOR_G_MINOR
Scale.B_MAJOR_A_FLAT_MINOR
Scale.C_MAJOR_A_MINOR
Scale.D_FLAT_MAJOR_B_FLAT_MINOR
Scale.D_MAJOR_B_MINOR
Scale.E_FLAT_MAJOR_C_MINOR
Scale.E_MAJOR_D_FLAT_MINOR
Scale.F_MAJOR_D_MINOR
Scale.G_FLAT_MAJOR_E_FLAT_MINOR
Scale.G_MAJOR_E_MINOR
Scale.SCALE_UNSPECIFIED
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 128/152
```
Schema
Schema.additional_properties
Schema.any_of
Schema.default
Schema.defs
Schema.description
Schema.enum
Schema.example
Schema.format
Schema.items
Schema.max_items
Schema.max_length
Schema.max_properties
Schema.maximum
Schema.min_items
Schema.min_length
Schema.min_properties
Schema.minimum
Schema.nullable
Schema.pattern
Schema.properties
Schema.property_ordering
Schema.ref
Schema.required
Schema.title
Schema.type
```
Schema.from_json_schema()
```
Schema.json_schema
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 129/152
```
SchemaDict
SchemaDict.additional_properties
SchemaDict.any_of
SchemaDict.default
SchemaDict.defs
SchemaDict.description
SchemaDict.enum
SchemaDict.example
SchemaDict.format
SchemaDict.max_items
SchemaDict.max_length
SchemaDict.max_properties
SchemaDict.maximum
SchemaDict.min_items
SchemaDict.min_length
SchemaDict.min_properties
SchemaDict.minimum
SchemaDict.nullable
SchemaDict.pattern
SchemaDict.properties
SchemaDict.property_ordering
SchemaDict.ref
SchemaDict.required
SchemaDict.title
SchemaDict.type
ScribbleImage
ScribbleImage.image
ScribbleImageDict
ScribbleImageDict.image
SearchEntryPoint
SearchEntryPoint.rendered_content
SearchEntryPoint.sdk_blob
SearchEntryPointDict
SearchEntryPointDict.rendered_content
SearchEntryPointDict.sdk_blob
SearchTypes
SearchTypes.image_search
SearchTypes.web_searchSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 130/152
```
SearchTypesDict
SearchTypesDict.image_search
SearchTypesDict.web_search
Segment
Segment.end_index
Segment.part_index
Segment.start_index
Segment.text
SegmentDict
SegmentDict.end_index
SegmentDict.part_index
SegmentDict.start_index
SegmentDict.text
SegmentImageConfig
SegmentImageConfig.binary_color_threshold
SegmentImageConfig.confidence_threshold
SegmentImageConfig.http_options
SegmentImageConfig.labels
SegmentImageConfig.mask_dilation
SegmentImageConfig.max_predictions
SegmentImageConfig.mode
SegmentImageConfigDict
SegmentImageConfigDict.binary_color_threshold
SegmentImageConfigDict.confidence_threshold
SegmentImageConfigDict.http_options
SegmentImageConfigDict.labels
SegmentImageConfigDict.mask_dilation
SegmentImageConfigDict.max_predictions
SegmentImageConfigDict.mode
SegmentImageResponse
SegmentImageResponse.generated_masks
SegmentImageResponseDict
SegmentImageResponseDict.generated_masks
SegmentImageSource
SegmentImageSource.image
SegmentImageSource.prompt
SegmentImageSource.scribble_image
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 131/152
```
SegmentImageSourceDict
SegmentImageSourceDict.image
SegmentImageSourceDict.prompt
SegmentImageSourceDict.scribble_image
SegmentMode
SegmentMode.BACKGROUND
SegmentMode.FOREGROUND
SegmentMode.INTERACTIVE
SegmentMode.PROMPT
SegmentMode.SEMANTIC
SessionResumptionConfig
SessionResumptionConfig.handle
SessionResumptionConfig.transparent
SessionResumptionConfigDict
SessionResumptionConfigDict.handle
SessionResumptionConfigDict.transparent
SingleEmbedContentResponse
SingleEmbedContentResponse.embedding
SingleEmbedContentResponse.token_count
SingleEmbedContentResponseDict
SingleEmbedContentResponseDict.embedding
SingleEmbedContentResponseDict.token_count
SlidingWindow
SlidingWindow.target_tokens
SlidingWindowDict
SlidingWindowDict.target_tokens
SpeakerVoiceConfig
SpeakerVoiceConfig.speaker
SpeakerVoiceConfig.voice_config
SpeakerVoiceConfigDict
SpeakerVoiceConfigDict.speaker
SpeakerVoiceConfigDict.voice_config
SpeechConfig
SpeechConfig.language_code
SpeechConfig.multi_speaker_voice_config
SpeechConfig.voice_config
SpeechConfigDict
SpeechConfigDict.language_code
SpeechConfigDict.multi_speaker_voice_config
SpeechConfigDict.voice_config
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 132/152
```
StartSensitivity
StartSensitivity.START_SENSITIVITY_HIGH
StartSensitivity.START_SENSITIVITY_LOW
StartSensitivity.START_SENSITIVITY_UNSPECIFIED
StreamableHttpTransport
StreamableHttpTransport.headers
StreamableHttpTransport.sse_read_timeout
StreamableHttpTransport.terminate_on_close
StreamableHttpTransport.timeout
StreamableHttpTransport.url
StreamableHttpTransportDict
StreamableHttpTransportDict.headers
StreamableHttpTransportDict.sse_read_timeout
StreamableHttpTransportDict.terminate_on_close
StreamableHttpTransportDict.timeout
StreamableHttpTransportDict.url
StringList
StringList.values
StringListDict
StyleReferenceConfig
StyleReferenceConfig.style_description
StyleReferenceConfigDict
StyleReferenceConfigDict.style_description
StyleReferenceImage
StyleReferenceImage.config
StyleReferenceImage.reference_id
StyleReferenceImage.reference_image
StyleReferenceImage.reference_type
StyleReferenceImage.style_image_config
StyleReferenceImageDict
StyleReferenceImageDict.config
StyleReferenceImageDict.reference_id
StyleReferenceImageDict.reference_image
StyleReferenceImageDict.reference_type
SubjectReferenceConfig
SubjectReferenceConfig.subject_description
SubjectReferenceConfig.subject_type
SubjectReferenceConfigDict
SubjectReferenceConfigDict.subject_description
SubjectReferenceConfigDict.subject_type
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 133/152
```
SubjectReferenceImage
SubjectReferenceImage.config
SubjectReferenceImage.reference_id
SubjectReferenceImage.reference_image
SubjectReferenceImage.reference_type
SubjectReferenceImage.subject_image_config
SubjectReferenceImageDict
SubjectReferenceImageDict.config
SubjectReferenceImageDict.reference_id
SubjectReferenceImageDict.reference_image
SubjectReferenceImageDict.reference_type
SubjectReferenceType
SubjectReferenceType.SUBJECT_TYPE_ANIMAL
SubjectReferenceType.SUBJECT_TYPE_DEFAULT
SubjectReferenceType.SUBJECT_TYPE_PERSON
SubjectReferenceType.SUBJECT_TYPE_PRODUCT
SupervisedHyperParameters
SupervisedHyperParameters.adapter_size
SupervisedHyperParameters.batch_size
SupervisedHyperParameters.epoch_count
SupervisedHyperParameters.learning_rate
SupervisedHyperParameters.learning_rate_multiplier
SupervisedHyperParametersDict
SupervisedHyperParametersDict.adapter_size
SupervisedHyperParametersDict.batch_size
SupervisedHyperParametersDict.epoch_count
SupervisedHyperParametersDict.learning_rate
SupervisedHyperParametersDict.learning_rate_multiplier
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 134/152
```
SupervisedTuningDataStats
SupervisedTuningDataStats.dropped_example_reasons
SupervisedTuningDataStats.total_billable_character_count
SupervisedTuningDataStats.total_billable_token_count
SupervisedTuningDataStats.total_truncated_example_count
SupervisedTuningDataStats.total_tuning_character_count
SupervisedTuningDataStats.truncated_example_indices
SupervisedTuningDataStats.tuning_dataset_example_count
SupervisedTuningDataStats.tuning_step_count
SupervisedTuningDataStats.user_dataset_examples
SupervisedTuningDataStats.user_input_token_distribution
SupervisedTuningDataStats.user_message_per_example_distribution
SupervisedTuningDataStats.user_output_token_distribution
SupervisedTuningDataStatsDict
SupervisedTuningDataStatsDict.dropped_example_reasons
SupervisedTuningDataStatsDict.total_billable_character_count
SupervisedTuningDataStatsDict.total_billable_token_count
SupervisedTuningDataStatsDict.total_truncated_example_count
SupervisedTuningDataStatsDict.total_tuning_character_count
SupervisedTuningDataStatsDict.truncated_example_indices
SupervisedTuningDataStatsDict.tuning_dataset_example_count
SupervisedTuningDataStatsDict.tuning_step_count
SupervisedTuningDataStatsDict.user_dataset_examples
SupervisedTuningDataStatsDict.user_input_token_distribution
SupervisedTuningDataStatsDict.user_message_per_example_distribution
SupervisedTuningDataStatsDict.user_output_token_distribution
SupervisedTuningDatasetDistribution
SupervisedTuningDatasetDistribution.billable_sum
SupervisedTuningDatasetDistribution.buckets
SupervisedTuningDatasetDistribution.max
SupervisedTuningDatasetDistribution.mean
SupervisedTuningDatasetDistribution.median
SupervisedTuningDatasetDistribution.min
SupervisedTuningDatasetDistribution.p5
SupervisedTuningDatasetDistribution.p95
SupervisedTuningDatasetDistribution.sum
SupervisedTuningDatasetDistributionDatasetBucket
SupervisedTuningDatasetDistributionDatasetBucket.count
SupervisedTuningDatasetDistributionDatasetBucket.left
SupervisedTuningDatasetDistributionDatasetBucket.right
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 135/152
```
SupervisedTuningDatasetDistributionDatasetBucketDict
SupervisedTuningDatasetDistributionDatasetBucketDict.count
SupervisedTuningDatasetDistributionDatasetBucketDict.left
SupervisedTuningDatasetDistributionDatasetBucketDict.right
SupervisedTuningDatasetDistributionDict
SupervisedTuningDatasetDistributionDict.billable_sum
SupervisedTuningDatasetDistributionDict.buckets
SupervisedTuningDatasetDistributionDict.max
SupervisedTuningDatasetDistributionDict.mean
SupervisedTuningDatasetDistributionDict.median
SupervisedTuningDatasetDistributionDict.min
SupervisedTuningDatasetDistributionDict.p5
SupervisedTuningDatasetDistributionDict.p95
SupervisedTuningDatasetDistributionDict.sum
SupervisedTuningSpec
SupervisedTuningSpec.export_last_checkpoint_only
SupervisedTuningSpec.hyper_parameters
SupervisedTuningSpec.training_dataset_uri
SupervisedTuningSpec.tuning_mode
SupervisedTuningSpec.validation_dataset_uri
SupervisedTuningSpecDict
SupervisedTuningSpecDict.export_last_checkpoint_only
SupervisedTuningSpecDict.hyper_parameters
SupervisedTuningSpecDict.training_dataset_uri
SupervisedTuningSpecDict.tuning_mode
SupervisedTuningSpecDict.validation_dataset_uri
TestTableFile
TestTableFile.comment
TestTableFile.parameter_names
TestTableFile.test_method
TestTableFile.test_table
TestTableFileDict
TestTableFileDict.comment
TestTableFileDict.parameter_names
TestTableFileDict.test_method
TestTableFileDict.test_table
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 136/152
```
TestTableItem
TestTableItem.exception_if_mldev
TestTableItem.exception_if_vertex
TestTableItem.has_union
TestTableItem.ignore_keys
TestTableItem.name
TestTableItem.override_replay_id
TestTableItem.parameters
TestTableItem.skip_in_api_mode
TestTableItemDict
TestTableItemDict.exception_if_mldev
TestTableItemDict.exception_if_vertex
TestTableItemDict.has_union
TestTableItemDict.ignore_keys
TestTableItemDict.name
TestTableItemDict.override_replay_id
TestTableItemDict.parameters
TestTableItemDict.skip_in_api_mode
ThinkingConfig
ThinkingConfig.include_thoughts
ThinkingConfig.thinking_budget
ThinkingConfig.thinking_level
ThinkingConfigDict
ThinkingConfigDict.include_thoughts
ThinkingConfigDict.thinking_budget
ThinkingConfigDict.thinking_level
ThinkingLevel
ThinkingLevel.HIGH
ThinkingLevel.LOW
ThinkingLevel.MEDIUM
ThinkingLevel.MINIMAL
ThinkingLevel.THINKING_LEVEL_UNSPECIFIED
TokensInfo
TokensInfo.role
TokensInfo.token_ids
TokensInfo.tokens
TokensInfoDict
TokensInfoDict.role
TokensInfoDict.token_ids
TokensInfoDict.tokens
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 137/152
```
Tool
Tool.code_execution
Tool.computer_use
Tool.enterprise_web_search
Tool.file_search
Tool.function_declarations
Tool.google_maps
Tool.google_search
Tool.google_search_retrieval
Tool.mcp_servers
Tool.retrieval
Tool.url_context
ToolCodeExecution
ToolCodeExecutionDict
ToolConfig
ToolConfig.function_calling_config
ToolConfig.retrieval_config
ToolConfigDict
ToolConfigDict.function_calling_config
ToolConfigDict.retrieval_config
ToolDict
ToolDict.code_execution
ToolDict.computer_use
ToolDict.enterprise_web_search
ToolDict.file_search
ToolDict.function_declarations
ToolDict.google_maps
ToolDict.google_search
ToolDict.google_search_retrieval
ToolDict.mcp_servers
ToolDict.retrieval
ToolDict.url_context
TrafficType
TrafficType.ON_DEMAND
TrafficType.PROVISIONED_THROUGHPUT
TrafficType.TRAFFIC_TYPE_UNSPECIFIED
Transcription
Transcription.finished
Transcription.text
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 138/152
```
TranscriptionDict
TranscriptionDict.finished
TranscriptionDict.text
TunedModel
TunedModel.checkpoints
TunedModel.endpoint
TunedModel.model
TunedModelCheckpoint
TunedModelCheckpoint.checkpoint_id
TunedModelCheckpoint.endpoint
TunedModelCheckpoint.epoch
TunedModelCheckpoint.step
TunedModelCheckpointDict
TunedModelCheckpointDict.checkpoint_id
TunedModelCheckpointDict.endpoint
TunedModelCheckpointDict.epoch
TunedModelCheckpointDict.step
TunedModelDict
TunedModelDict.checkpoints
TunedModelDict.endpoint
TunedModelDict.model
TunedModelInfo
TunedModelInfo.base_model
TunedModelInfo.create_time
TunedModelInfo.update_time
TunedModelInfoDict
TunedModelInfoDict.base_model
TunedModelInfoDict.create_time
TunedModelInfoDict.update_time
TuningDataStats
TuningDataStats.distillation_data_stats
TuningDataStats.preference_optimization_data_stats
TuningDataStats.supervised_tuning_data_stats
TuningDataStatsDict
TuningDataStatsDict.distillation_data_stats
TuningDataStatsDict.preference_optimization_data_stats
TuningDataStatsDict.supervised_tuning_data_stats
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 139/152
```
TuningDataset
TuningDataset.examples
TuningDataset.gcs_uri
TuningDataset.vertex_dataset_resource
TuningDatasetDict
TuningDatasetDict.examples
TuningDatasetDict.gcs_uri
TuningDatasetDict.vertex_dataset_resource
TuningExample
TuningExample.output
TuningExample.text_input
TuningExampleDict
TuningExampleDict.output
TuningExampleDict.text_input
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 140/152
```
TuningJob
TuningJob.base_model
TuningJob.create_time
TuningJob.custom_base_model
TuningJob.description
TuningJob.distillation_spec
TuningJob.encryption_spec
TuningJob.end_time
TuningJob.error
TuningJob.evaluation_config
TuningJob.experiment
TuningJob.labels
TuningJob.name
TuningJob.output_uri
TuningJob.partner_model_tuning_spec
TuningJob.pipeline_job
TuningJob.pre_tuned_model
TuningJob.preference_optimization_spec
TuningJob.sdk_http_response
TuningJob.service_account
TuningJob.start_time
TuningJob.state
TuningJob.supervised_tuning_spec
TuningJob.tuned_model
TuningJob.tuned_model_display_name
TuningJob.tuning_data_stats
TuningJob.update_time
TuningJob.veo_tuning_spec
TuningJob.has_ended
TuningJob.has_succeeded
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 141/152
```
TuningJobDict
TuningJobDict.base_model
TuningJobDict.create_time
TuningJobDict.custom_base_model
TuningJobDict.description
TuningJobDict.distillation_spec
TuningJobDict.encryption_spec
TuningJobDict.end_time
TuningJobDict.error
TuningJobDict.evaluation_config
TuningJobDict.experiment
TuningJobDict.labels
TuningJobDict.name
TuningJobDict.output_uri
TuningJobDict.partner_model_tuning_spec
TuningJobDict.pipeline_job
TuningJobDict.pre_tuned_model
TuningJobDict.preference_optimization_spec
TuningJobDict.sdk_http_response
TuningJobDict.service_account
TuningJobDict.start_time
TuningJobDict.state
TuningJobDict.supervised_tuning_spec
TuningJobDict.tuned_model
TuningJobDict.tuned_model_display_name
TuningJobDict.tuning_data_stats
TuningJobDict.update_time
TuningJobDict.veo_tuning_spec
TuningMethod
TuningMethod.DISTILLATION
TuningMethod.PREFERENCE_TUNING
TuningMethod.SUPERVISED_FINE_TUNING
TuningMode
TuningMode.TUNING_MODE_FULL
TuningMode.TUNING_MODE_PEFT_ADAPTER
TuningMode.TUNING_MODE_UNSPECIFIED
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 142/152
```
TuningOperation
TuningOperation.done
TuningOperation.error
TuningOperation.metadata
TuningOperation.name
TuningOperation.sdk_http_response
TuningOperationDict
TuningOperationDict.done
TuningOperationDict.error
TuningOperationDict.metadata
TuningOperationDict.name
TuningOperationDict.sdk_http_response
TuningTask
TuningTask.TUNING_TASK_I2V
TuningTask.TUNING_TASK_R2V
TuningTask.TUNING_TASK_T2V
TuningTask.TUNING_TASK_UNSPECIFIED
TuningValidationDataset
TuningValidationDataset.gcs_uri
TuningValidationDataset.vertex_dataset_resource
TuningValidationDatasetDict
TuningValidationDatasetDict.gcs_uri
TuningValidationDatasetDict.vertex_dataset_resource
TurnCompleteReason
TurnCompleteReason.MALFORMED_FUNCTION_CALL
TurnCompleteReason.NEED_MORE_INPUT
TurnCompleteReason.RESPONSE_REJECTED
TurnCompleteReason.TURN_COMPLETE_REASON_UNSPECIFIED
TurnCoverage
TurnCoverage.TURN_COVERAGE_UNSPECIFIED
TurnCoverage.TURN_INCLUDES_ALL_INPUT
TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 143/152
```
Type
Type.ARRAY
Type.BOOLEAN
Type.INTEGER
Type.NULL
Type.NUMBER
Type.OBJECT
Type.STRING
Type.TYPE_UNSPECIFIED
UnifiedMetric
UnifiedMetric.bleu_spec
UnifiedMetric.computation_based_metric_spec
UnifiedMetric.custom_code_execution_spec
UnifiedMetric.llm_based_metric_spec
UnifiedMetric.pointwise_metric_spec
UnifiedMetric.predefined_metric_spec
UnifiedMetric.rouge_spec
UnifiedMetricDict
UnifiedMetricDict.bleu_spec
UnifiedMetricDict.computation_based_metric_spec
UnifiedMetricDict.custom_code_execution_spec
UnifiedMetricDict.llm_based_metric_spec
UnifiedMetricDict.pointwise_metric_spec
UnifiedMetricDict.predefined_metric_spec
UnifiedMetricDict.rouge_spec
UpdateCachedContentConfig
UpdateCachedContentConfig.expire_time
UpdateCachedContentConfig.http_options
UpdateCachedContentConfig.ttl
UpdateCachedContentConfigDict
UpdateCachedContentConfigDict.expire_time
UpdateCachedContentConfigDict.http_options
UpdateCachedContentConfigDict.ttl
UpdateModelConfig
UpdateModelConfig.default_checkpoint_id
UpdateModelConfig.description
UpdateModelConfig.display_name
UpdateModelConfig.http_optionsSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 144/152
```
UpdateModelConfigDict
UpdateModelConfigDict.default_checkpoint_id
UpdateModelConfigDict.description
UpdateModelConfigDict.display_name
UpdateModelConfigDict.http_options
UploadFileConfig
UploadFileConfig.display_name
UploadFileConfig.http_options
UploadFileConfig.mime_type
UploadFileConfig.name
UploadFileConfigDict
UploadFileConfigDict.display_name
UploadFileConfigDict.http_options
UploadFileConfigDict.mime_type
UploadFileConfigDict.name
UploadToFileSearchStoreConfig
UploadToFileSearchStoreConfig.chunking_config
UploadToFileSearchStoreConfig.custom_metadata
UploadToFileSearchStoreConfig.display_name
UploadToFileSearchStoreConfig.http_options
UploadToFileSearchStoreConfig.mime_type
UploadToFileSearchStoreConfig.should_return_http_response
UploadToFileSearchStoreConfigDict
UploadToFileSearchStoreConfigDict.chunking_config
UploadToFileSearchStoreConfigDict.custom_metadata
UploadToFileSearchStoreConfigDict.display_name
UploadToFileSearchStoreConfigDict.http_options
UploadToFileSearchStoreConfigDict.mime_type
UploadToFileSearchStoreConfigDict.should_return_http_response
UploadToFileSearchStoreOperation
UploadToFileSearchStoreOperation.response
```
UploadToFileSearchStoreOperation.from_api_response()
```
UploadToFileSearchStoreResponse
UploadToFileSearchStoreResponse.document_name
UploadToFileSearchStoreResponse.parent
UploadToFileSearchStoreResponse.sdk_http_response
UploadToFileSearchStoreResponseDict
UploadToFileSearchStoreResponseDict.document_name
UploadToFileSearchStoreResponseDict.parent
UploadToFileSearchStoreResponseDict.sdk_http_response
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 145/152
```
UploadToFileSearchStoreResumableResponse
UploadToFileSearchStoreResumableResponse.sdk_http_response
UploadToFileSearchStoreResumableResponseDict
UploadToFileSearchStoreResumableResponseDict.sdk_http_response
UpscaleImageConfig
UpscaleImageConfig.enhance_input_image
UpscaleImageConfig.http_options
UpscaleImageConfig.image_preservation_factor
UpscaleImageConfig.include_rai_reason
UpscaleImageConfig.labels
UpscaleImageConfig.output_compression_quality
UpscaleImageConfig.output_gcs_uri
UpscaleImageConfig.output_mime_type
UpscaleImageConfig.person_generation
UpscaleImageConfig.safety_filter_level
UpscaleImageConfigDict
UpscaleImageConfigDict.enhance_input_image
UpscaleImageConfigDict.http_options
UpscaleImageConfigDict.image_preservation_factor
UpscaleImageConfigDict.include_rai_reason
UpscaleImageConfigDict.labels
UpscaleImageConfigDict.output_compression_quality
UpscaleImageConfigDict.output_gcs_uri
UpscaleImageConfigDict.output_mime_type
UpscaleImageConfigDict.person_generation
UpscaleImageConfigDict.safety_filter_level
UpscaleImageParameters
UpscaleImageParameters.config
UpscaleImageParameters.image
UpscaleImageParameters.model
UpscaleImageParameters.upscale_factor
UpscaleImageParametersDict
UpscaleImageParametersDict.config
UpscaleImageParametersDict.image
UpscaleImageParametersDict.model
UpscaleImageParametersDict.upscale_factor
UpscaleImageResponse
UpscaleImageResponse.generated_images
UpscaleImageResponse.sdk_http_response
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 146/152
```
UpscaleImageResponseDict
UpscaleImageResponseDict.generated_images
UpscaleImageResponseDict.sdk_http_response
UrlContext
UrlContextDict
UrlContextMetadata
UrlContextMetadata.url_metadata
UrlContextMetadataDict
UrlContextMetadataDict.url_metadata
UrlMetadata
UrlMetadata.retrieved_url
UrlMetadata.url_retrieval_status
UrlMetadataDict
UrlMetadataDict.retrieved_url
UrlMetadataDict.url_retrieval_status
UrlRetrievalStatus
UrlRetrievalStatus.URL_RETRIEVAL_STATUS_ERROR
UrlRetrievalStatus.URL_RETRIEVAL_STATUS_PAYWALL
UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS
UrlRetrievalStatus.URL_RETRIEVAL_STATUS_UNSAFE
UrlRetrievalStatus.URL_RETRIEVAL_STATUS_UNSPECIFIED
UsageMetadata
UsageMetadata.cache_tokens_details
UsageMetadata.cached_content_token_count
UsageMetadata.prompt_token_count
UsageMetadata.prompt_tokens_details
UsageMetadata.response_token_count
UsageMetadata.response_tokens_details
UsageMetadata.thoughts_token_count
UsageMetadata.tool_use_prompt_token_count
UsageMetadata.tool_use_prompt_tokens_details
UsageMetadata.total_token_count
UsageMetadata.traffic_type
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 147/152
```
UsageMetadataDict
UsageMetadataDict.cache_tokens_details
UsageMetadataDict.cached_content_token_count
UsageMetadataDict.prompt_token_count
UsageMetadataDict.prompt_tokens_details
UsageMetadataDict.response_token_count
UsageMetadataDict.response_tokens_details
UsageMetadataDict.thoughts_token_count
UsageMetadataDict.tool_use_prompt_token_count
UsageMetadataDict.tool_use_prompt_tokens_details
UsageMetadataDict.total_token_count
UsageMetadataDict.traffic_type
UserContent
UserContent.parts
UserContent.role
VadSignalType
VadSignalType.VAD_SIGNAL_TYPE_EOS
VadSignalType.VAD_SIGNAL_TYPE_SOS
VadSignalType.VAD_SIGNAL_TYPE_UNSPECIFIED
VeoHyperParameters
VeoHyperParameters.epoch_count
VeoHyperParameters.learning_rate_multiplier
VeoHyperParameters.tuning_task
VeoHyperParametersDict
VeoHyperParametersDict.epoch_count
VeoHyperParametersDict.learning_rate_multiplier
VeoHyperParametersDict.tuning_task
VeoTuningSpec
VeoTuningSpec.hyper_parameters
VeoTuningSpec.training_dataset_uri
VeoTuningSpec.validation_dataset_uri
VeoTuningSpecDict
VeoTuningSpecDict.hyper_parameters
VeoTuningSpecDict.training_dataset_uri
VeoTuningSpecDict.validation_dataset_uri
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 148/152
```
VertexAISearch
VertexAISearch.data_store_specs
VertexAISearch.datastore
VertexAISearch.engine
VertexAISearch.filter
VertexAISearch.max_results
VertexAISearchDataStoreSpec
VertexAISearchDataStoreSpec.data_store
VertexAISearchDataStoreSpec.filter
VertexAISearchDataStoreSpecDict
VertexAISearchDataStoreSpecDict.data_store
VertexAISearchDataStoreSpecDict.filter
VertexAISearchDict
VertexAISearchDict.data_store_specs
VertexAISearchDict.datastore
VertexAISearchDict.engine
VertexAISearchDict.filter
VertexAISearchDict.max_results
VertexRagStore
VertexRagStore.rag_corpora
VertexRagStore.rag_resources
VertexRagStore.rag_retrieval_config
VertexRagStore.similarity_top_k
VertexRagStore.store_context
VertexRagStore.vector_distance_threshold
VertexRagStoreDict
VertexRagStoreDict.rag_corpora
VertexRagStoreDict.rag_resources
VertexRagStoreDict.rag_retrieval_config
VertexRagStoreDict.similarity_top_k
VertexRagStoreDict.store_context
VertexRagStoreDict.vector_distance_threshold
VertexRagStoreRagResource
VertexRagStoreRagResource.rag_corpus
VertexRagStoreRagResource.rag_file_ids
VertexRagStoreRagResourceDict
VertexRagStoreRagResourceDict.rag_corpus
VertexRagStoreRagResourceDict.rag_file_idsSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 149/152
```
Video
Video.mime_type
Video.uri
Video.video_bytes
```
Video.from_file()
```
```
Video.save()
```
```
Video.show()
```
VideoCompressionQuality
VideoCompressionQuality.LOSSLESS
VideoCompressionQuality.OPTIMIZED
VideoDict
VideoDict.mime_type
VideoDict.uri
VideoDict.video_bytes
VideoGenerationMask
VideoGenerationMask.image
VideoGenerationMask.mask_mode
VideoGenerationMaskDict
VideoGenerationMaskDict.image
VideoGenerationMaskDict.mask_mode
VideoGenerationMaskMode
VideoGenerationMaskMode.INSERT
VideoGenerationMaskMode.OUTPAINT
VideoGenerationMaskMode.REMOVE
VideoGenerationMaskMode.REMOVE_STATIC
VideoGenerationReferenceImage
VideoGenerationReferenceImage.image
VideoGenerationReferenceImage.reference_type
VideoGenerationReferenceImageDict
VideoGenerationReferenceImageDict.image
VideoGenerationReferenceImageDict.reference_type
VideoGenerationReferenceType
VideoGenerationReferenceType.ASSET
VideoGenerationReferenceType.STYLE
VideoMetadata
VideoMetadata.end_offset
VideoMetadata.fps
VideoMetadata.start_offsetSkip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 150/152
```
VideoMetadataDict
VideoMetadataDict.end_offset
VideoMetadataDict.fps
VideoMetadataDict.start_offset
VoiceActivity
VoiceActivity.voice_activity_type
VoiceActivityDetectionSignal
VoiceActivityDetectionSignal.vad_signal_type
VoiceActivityDetectionSignalDict
VoiceActivityDetectionSignalDict.vad_signal_type
VoiceActivityDict
VoiceActivityDict.voice_activity_type
VoiceActivityType
VoiceActivityType.ACTIVITY_END
VoiceActivityType.ACTIVITY_START
VoiceActivityType.TYPE_UNSPECIFIED
VoiceConfig
VoiceConfig.prebuilt_voice_config
VoiceConfig.replicated_voice_config
VoiceConfigDict
VoiceConfigDict.prebuilt_voice_config
VoiceConfigDict.replicated_voice_config
WebSearch
WebSearchDict
WeightedPrompt
WeightedPrompt.text
WeightedPrompt.weight
WeightedPromptDict
WeightedPromptDict.text
WeightedPromptDict.weight
WhiteSpaceConfig
WhiteSpaceConfig.max_overlap_tokens
WhiteSpaceConfig.max_tokens_per_chunk
WhiteSpaceConfigDict
WhiteSpaceConfigDict.max_overlap_tokens
WhiteSpaceConfigDict.max_tokens_per_chunk
Skip to content
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 151/152
```
Copyright © 2024, Google
Made with Sphinx and @pradyunsg's Furo
3/13/26, 1:55 AM Google Gen AI SDK documentation
```
https://googleapis.github.io/python-genai/ 152/152
```