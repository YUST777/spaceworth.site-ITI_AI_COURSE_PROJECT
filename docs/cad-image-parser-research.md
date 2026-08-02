# CAD image parser decision

## Selected baseline

Use [`Yytsi/floorplan-to-3d`](https://github.com/Yytsi/floorplan-to-3d) as the first integration baseline.

- MIT licensed.
- Includes a FastAPI server and Docker workflow.
- Uses a pretrained U-Net wall/opening segmentation model.
- Publishes a roughly 98 MB model checkpoint on Hugging Face.
- Produces structured wall and opening geometry that can be serialized as JSON.
- Runs on CPU or CUDA, so it is realistic to benchmark as a separate Railway service.

The upstream server currently accepts SVG uploads. Our adapter must also accept PNG, JPG and WEBP by passing the decoded raster image into the extractor instead of requiring SVG rendering.

## Output contract

The parser service should return geometry only. The existing 90.64% R² price model does not understand images directly.

```json
{
  "image_width": 1600,
  "image_height": 1200,
  "walls": [
    { "polygon": [[120, 90], [860, 90], [860, 112], [120, 112]] }
  ],
  "openings": [
    { "type": "door", "polygon": [[410, 90], [475, 90], [475, 112], [410, 112]] }
  ],
  "rooms": [],
  "confidence": 0.86
}
```

The frontend can render this geometry and combine it with the user's bedrooms, bathrooms, area and location before calling the existing Railway price API.

## Deployment plan

1. Keep the current price API as its own Railway service.
2. Create a separate `cad-parser-api` service with a strict upload limit and request rate limit.
3. Download the parser checkpoint during the image build, not on every request.
4. Benchmark cold start, peak RAM and one real PNG before enabling the upload button in production.
5. Connect `VITE_PLAN_ANALYSIS_API_URL` only after the parser returns real geometry.

## Heavier alternative

[`Cornell-VAILab/Raster2Seq`](https://github.com/Cornell-VAILab/Raster2Seq) has stronger room/icon polygon output and an MIT license, but its published checkpoint is about 1.45 GB and its documented environment uses CUDA 12.1. It is not the right first target for the current Railway trial.
