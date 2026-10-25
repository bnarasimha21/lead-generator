# 🎤 AMD GPU Droplet Demo

A modern FastAPI web app that lets users record their requirements via audio, transcribes them using OpenAI Whisper, and uses an LLM to extract user, company, query, and action details. Results are displayed instantly in a beautiful UI.

---

## Features
- Record audio in-browser (no install needed)
- Transcribe audio to text using OpenAI Whisper
- Extract structured info (user, company, query, action) using `gpt-3.5-turbo`. The reply is parsed as JSON; if the model wraps it in prose, the outermost `{...}` block is used, and if that is not valid JSON the page shows "Extraction failed" with the error. Missing keys come back blank.
- Modern, responsive UI
- Nothing is kept by the app: the upload is written to a uniquely named temp file, sent to OpenAI for transcription, and deleted straight away. The audio and transcript do go to OpenAI, so this is not an on-device or private-by-design setup

---

## Local Development

### 1. Clone the repo
```bash
git clone https://github.com/bnarasimha/Lead-Generator.git
cd Lead-Generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY=sk-...
```

### 4. Run the app
```bash
uvicorn app.main:app --reload
```

Visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## Deployment: DigitalOcean App Platform

1. Push your code to GitHub.
2. Create a new app on [DigitalOcean App Platform](https://cloud.digitalocean.com/apps).
3. Connect your repo and select the Python environment.
4. Set the run command:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```
5. Add the environment variable:
   - `OPENAI_API_KEY` (as a secret)
6. Deploy and access your public URL!

---

## Environment Variables
- `OPENAI_API_KEY`: your OpenAI API key for Whisper and GPT. Without it, both calls fail and the page shows the error text.

---

## Example Prompt
> "Hi, I am John from XYZ company, I am trying to use GPUs for my AI/ML workloads. Can you help me?"

---

## License
MIT 