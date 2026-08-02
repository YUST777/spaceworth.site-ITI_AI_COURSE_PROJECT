# CAD image analysis decision

## Live implementation

SpaceWorth uses Gemini 3.5 Flash-Lite as the first production implementation for uploaded floor plans and CAD drawings. The existing Railway FastAPI service accepts PNG, JPG, WEBP and PDF files through `POST /analyze`, sends the file to Gemini and validates the response against a structured Pydantic schema before using any extracted value.

Gemini is responsible for reading information that is visible in the drawing:

- room labels and room types;
- bedroom, bathroom, balcony and parking counts;
- printed dimensions and dimension text;
- total area only when it is explicitly printed or can be calculated from reliable printed dimensions;
- per-room confidence, overall confidence and warnings.

The service never converts image pixels into square footage. When a drawing has no scale or trustworthy printed dimensions, `total_area_sqft` remains `null` and the existing property area is retained.

## Live response contract

The analysis response includes the validated CAD interpretation and the result from the price model in one request. A representative shape is:

```json
{
  "analysis_id": "cad_01K...",
  "query_id": "7e0e...",
  "predicted_price_inr": 7303740.70,
  "vision_model": "gemini-3.5-flash-lite",
  "analysis": {
    "usable": true,
    "property_type": "flat",
    "bedrooms": 2,
    "bathrooms": 2,
    "balconies": 1,
    "parking_spaces": null,
    "total_area_sqft": 1200,
    "area_source": "printed_total",
    "rooms": [
      {
        "label": "Master Bedroom",
        "category": "bedroom",
        "dimensions": "14 ft x 12 ft",
        "area_sqft": 168,
        "confidence": 0.94
      }
    ],
    "warnings": [],
    "confidence": 0.95
  }
}
```

## Price-model boundary

The held-out 90.64% R2 tabular ensemble remains separate from Gemini. Gemini does not estimate the property price. The API merges only validated extracted property fields into the normal `PropertyInput`, runs the same existing ensemble used by `POST /predict` and stores both the CAD trace and prediction in PostgreSQL.

This separation keeps image interpretation auditable and preserves the existing model's evaluation claim. Users can still edit their normal property details before starting an analysis, and those values act as fallbacks when the drawing cannot supply a field reliably.

## Open-source segmentation later

An open-source floor-plan segmentation model may be added later to draw wall, door and window overlays as extra visual evidence. [`Yytsi/floorplan-to-3d`](https://github.com/Yytsi/floorplan-to-3d) remains a possible benchmark because it is MIT licensed, has a relatively small checkpoint and can run on CPU or CUDA.

That future component would not replace Gemini's OCR and label extraction, and it would not feed pixel-derived area into the price model. SpaceWorth does not currently claim image-to-geometry reconstruction, wall polygons, door polygons, window polygons or generated 3D output as live features.
