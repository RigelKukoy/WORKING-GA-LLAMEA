import os
from dotenv import load_dotenv
from google import genai

def main():
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    quota_project = os.getenv("GOOGLE_CLOUD_QUOTA_PROJECT")
    
    if quota_project:
        os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = quota_project
    
    print(f"Project: {project}, Location: {location}, Quota: {quota_project}")
    
    client = genai.Client(vertexai=True, project=project, location=location)
    
    print("Sending content generation request...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Tell me one interesting fact about polymers.'
        )
        print("\nResponse:")
        print(response.text)
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
