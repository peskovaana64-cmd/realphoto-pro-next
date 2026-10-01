# DEPLOY NEW PUBLIC SITE

Use a NEW GitHub repository so the old Streamlit URL stays untouched.

Recommended repository name:
`realphoto-pro-next`

Recommended public Streamlit URL:
`realphoto-pro-next.streamlit.app`
(or any available name Streamlit offers)

## Upload
Upload ALL files from this project root into the new repository.

Main file path:
`streamlit_app.py`

## Streamlit Community Cloud
1. Open https://share.streamlit.io/
2. New app
3. Repository: `<your-account>/realphoto-pro-next`
4. Branch: `main`
5. Main file path: `streamlit_app.py`
6. Choose a NEW app URL.
7. Deploy.

Do not edit or disconnect the old app/repository.
That preserves the old public link.

## Secrets later
In Streamlit app settings -> Secrets, tech team can add:

OPENAI_API_KEY="..."
GEMINI_API_KEY="..."
MAGNIFIC_API_KEY="..."

GPU/local AI modules are separate from the public CPU Streamlit version.
