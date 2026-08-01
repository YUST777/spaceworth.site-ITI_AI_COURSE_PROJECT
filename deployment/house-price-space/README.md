---
title: ITI House Price API
emoji: 🏠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# ITI House Price API

FastAPI service for the final house-price ensemble. It serves `POST /predict` and `GET /health` on port `7860`.

## Example request

```json
{
  "area_sqft": 1200,
  "area_type": "super",
  "location": "thane",
  "locality": "kolshet_road",
  "society": "lodha_amara",
  "bedrooms": 2,
  "bathroom": 2,
  "floor_num": 8,
  "total_floors": 24,
  "furnishing": "semi_furnished"
}
```

Set `ALLOWED_ORIGINS` to the production Vercel domain after deploying the frontend. Set `API_KEY` only if requests are proxied through a server-side Vercel route; never expose that key in browser code.
