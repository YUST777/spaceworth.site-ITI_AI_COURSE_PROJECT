# Free Render deployment

This deploys the validated 90.64% R² house-price API as a free Render web service. The Docker build downloads the public model pieces from the Hugging Face model repository and reconstructs the real `house_price.pkl` model; no model retraining or substitute model is used.

## Deploy

1. Sign in to Render with GitHub and open the repository's **Blueprint** deployment flow.
2. Select `render.yaml` from this repository.
3. Create the free `iti-house-price-api` web service.
4. When Render shows the service URL, set `ALLOWED_ORIGINS` to the Vercel production URL and preview URL pattern if desired.
5. Set a long random `API_KEY` environment variable, then send it in the frontend as the `X-API-Key` request header.

The API exposes `GET /health` and `POST /predict`. Free services sleep after inactivity, so the first request after sleep can take longer while Render starts the container.
