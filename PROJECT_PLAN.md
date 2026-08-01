# AI House Intelligence and CAD Platform - Master Project Plan

## 1. Product Vision

Build a complete machine-learning web product that predicts Indian house prices and presents the result through a premium property-planning interface inspired by the supplied SpaceMap reference.

The application is not just a price form. It combines four connected capabilities:

1. House-price prediction from real property data.
2. AI-assisted CAD generation from property requirements.
3. AI-assisted CAD extraction from uploaded floor-plan images.
4. Multi-property and neighborhood comparison through an interactive map.

The final experience should feel like a real property intelligence and planning product rather than a classroom demo.

## 2. Non-Negotiable Requirements

- Complete the assignment requirements in `House_Price_Prediction_Project_Guide.pdf`.
- Train and evaluate at least two regression models on the supplied dataset.
- Report test-set MAE, RMSE, and R2 and justify the selected model.
- Export a reproducible preprocessing and model pipeline.
- Serve real model predictions through FastAPI.
- Build the frontend with React, TypeScript, and Vite.
- Use TanStack libraries where they improve routing, data fetching, forms, or tables.
- Use shadcn/ui or similarly polished modern primitives.
- Follow the supplied SpaceMap screenshot closely for structure and visual direction.
- Support multiple houses or units in one project.
- Include a map-based neighborhood evaluation experience.
- Provide real loading, validation, empty, error, and success states.
- Do not use mock predictions in the final application.
- Keep secrets, `.env` files, the virtual environment, dependencies, and the raw CSV out of Git.
- Maintain a clean Git history and publish the completed project to `YUST777/iti_ai_project`.
- Deploy the frontend and backend and verify the public end-to-end flow.

## 3. Current Repository State

The repository currently contains:

- The assignment PDF.
- The raw `house_prices.csv` dataset.
- `.gitignore`.
- `notebooks/pipeline.py`, containing shared parsing, feature engineering, outlier filtering, and deduplication logic.

Still required:

- Training notebook.
- Model comparison and exported model.
- Allowed-locations artifact.
- FastAPI backend and tests.
- React frontend.
- CAD scene model and editor.
- AI CAD drawer.
- AI CAD extractor.
- Map and neighborhood evaluation.
- Documentation, screenshots, deployment, and end-to-end verification.

## 4. Verified Dataset Facts

- 187,531 rows.
- 21 source columns.
- 81 locations.
- `Dimensions` and `Plot Area` are entirely empty.
- `Society`, `Super Area`, and `Car Parking` contain substantial missing data.
- The current cleaning pipeline produces approximately 59,828 filtered and deduplicated training rows.
- The target price is heavily skewed and should be examined on a logarithmic scale.
- A GPU is not required for this tabular regression problem. CPU-based scikit-learn models are sufficient.

## 5. Reference UI Structure

The primary desktop workspace should closely follow the supplied reference image.

### 5.1 Top Header

- Product logo and name.
- Project or property name.
- List View / Visual View switch.
- Save action.
- Export or Publish action.
- User/project status when useful.

### 5.2 Left Navigation Rail

- Home/project overview.
- Properties or units.
- CAD workspace.
- Neighborhood/map analysis.
- Model insights or analytics.
- Settings.

### 5.3 Left Property Panel

- Basic Info and More Details tabs.
- Property title and optional description.
- Location selector.
- Carpet/super area and unit selector.
- Bedrooms and bathrooms.
- Balconies and parking.
- Current floor and total floors.
- Property type.
- Furnishing.
- Transaction type.
- Ownership.
- Facing and overlooking.
- Optional natural-language planning instructions.
- Generate/Add to Canvas action.

The current price and price per square foot must not be model inputs because they reveal the prediction target. They can be displayed only as model output or comparison data.

### 5.4 Central Workspace

- Large interactive canvas.
- 2D Plan and 3D View tabs.
- Pan, zoom, fit-to-screen, undo, and redo.
- Select, wall, door, window, room, label, plant/furniture, and delete tools.
- Dimension lines and room-area labels.
- Generation/extraction progress state.
- Confidence and warning overlays when AI output is uncertain.

### 5.5 Right Summary Panel

- Predicted property price.
- Estimated price per square foot.
- Prediction range or confidence information when defensible.
- Complete property summary.
- Neighborhood comparison.
- Multiple-unit cards with thumbnails.
- Add, duplicate, compare, remove, and select-unit actions.

### 5.6 Responsive Behavior

- Desktop retains the three-column professional workspace.
- Tablet collapses the right panel into a drawer.
- Mobile prioritizes the form, results, and unit list; the CAD editor may use a dedicated full-screen route.

## 6. Core User Flows

### 6.1 Property Price Prediction

1. User creates a project or unit.
2. User enters property details.
3. Client validates the inputs.
4. Frontend submits the exact model schema to FastAPI.
5. Backend applies the same preprocessing used during training.
6. Model returns a real predicted price.
7. UI formats the result in rupees, Lac, or Crore.
8. Result, inputs, CAD plan, and map context remain attached to that unit.

### 6.2 AI CAD Drawer - Information to Editable Plan

The user supplies structured fields and optional natural-language instructions, for example:

> Create a 1,500 sqft north-facing home with three bedrooms, two bathrooms, an open kitchen, one balcony, and parking.

The system should:

1. Convert the request into a validated layout constraint object.
2. Generate a plausible initial room arrangement.
3. Produce walls, doors, windows, room polygons, labels, and dimensions.
4. Check room overlap, boundary violations, minimum sizes, and circulation.
5. Render the result as an editable 2D vector/CAD scene.
6. Allow the user to manually edit every generated element.
7. Extrude the same scene into an optional 3D preview.

The reliable baseline should be a procedural constraint-based layout engine. An AI or language model may translate natural language into structured constraints and propose layout variations, but the geometry must remain deterministic, validated, and editable.

### 6.3 AI CAD Extractor - Image to Editable Plan

Supported high-confidence sources:

- Digital floor-plan image.
- Blueprint.
- Scanned plan.
- Hand-drawn plan.
- Phone photograph of a printed floor plan.

The extraction pipeline should:

1. Upload and safely validate the image.
2. Correct perspective, rotation, contrast, and noise.
3. Detect the outer boundary and internal wall lines.
4. Detect doors, windows, room regions, symbols, and dimension lines.
5. Use OCR for room names, dimensions, and annotations.
6. Infer scale when a known measurement exists.
7. Convert detected geometry into the same editable CAD scene format used by the drawer.
8. Show confidence and highlight uncertain elements for user confirmation.
9. Allow manual correction before saving.

An ordinary single interior or exterior photograph cannot reliably reconstruct a complete building. That mode must be labeled as an estimated concept unless the user supplies multiple images, video, LiDAR/depth information, or measurements.

### 6.4 Multiple Properties or Units

- A project can contain several houses, apartments, or alternative plans.
- Each unit stores its own form values, prediction, CAD scene, map location, and notes.
- Users can duplicate a unit and change one variable for comparison.
- The comparison view should show price, area, price/sqft, room counts, location, and prediction differences.

### 6.5 Neighborhood and Map Evaluation

- Search for a place or select a location on the map.
- Show the selected unit and comparable properties as markers.
- Display location-level model statistics derived from the dataset.
- Use price tiers or a heatmap where the data supports them.
- Show nearby services and useful geographic context when a maps provider supports it.
- Keep map-provider calls behind an adapter so Google Maps can be replaced if credentials, billing, or limits become a blocker.
- Never invent live comparable listings. Clearly distinguish dataset-derived statistics from live external data.

## 7. Technical Architecture

### 7.1 Machine Learning

- Python 3.11+.
- pandas and NumPy.
- scikit-learn `Pipeline` and `ColumnTransformer`.
- Baseline model such as Linear Regression or HistGradientBoosting.
- Stronger comparison model such as Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, or a suitable open-source tabular model.
- joblib model export.
- Reproducible random seeds and pinned versions.

### 7.2 Backend

- FastAPI.
- Pydantic request/response schemas.
- Model loaded once through application lifespan.
- Shared preprocessing contract.
- Clear service boundaries for prediction, CAD generation, CAD extraction, mapping, and persistence.
- Structured logging and centralized error handling.
- CORS configured through environment settings.
- pytest and FastAPI `TestClient`/HTTPX tests.

Suggested API surface:

- `GET /health`
- `GET /model/metadata`
- `POST /predict`
- `POST /cad/generate`
- `POST /cad/extract`
- `POST /cad/validate`
- `GET /locations`
- `GET /neighborhood/{location}`
- Project and unit CRUD endpoints if server persistence is included.

### 7.3 Frontend

- React + TypeScript + Vite.
- TanStack Query for server state.
- TanStack Router or React Router for routes.
- React Hook Form plus schema validation, or TanStack Form.
- shadcn/ui primitives.
- Tailwind CSS.
- Zustand or an equivalent focused store for CAD editor state.
- SVG or Canvas-based 2D editor.
- Three.js/React Three Fiber for optional 3D extrusion.
- Map adapter using Google Maps, MapLibre, or Leaflet depending on final provider choice.

### 7.4 Shared CAD Scene Format

Both AI-CAD workflows must produce the same versioned scene representation:

- Project and unit metadata.
- Canvas dimensions and scale.
- Exterior boundary.
- Rooms as polygons.
- Walls as line segments with thickness.
- Doors and windows attached to walls.
- Labels and measurements.
- Furniture/symbol objects.
- Source type: generated, extracted, or manual.
- Confidence values and unresolved warnings.

This shared format allows editing, saving, comparison, SVG/PDF export, and 3D extrusion without maintaining separate systems.

## 8. Target Repository Structure

```text
iti_project/
|-- notebooks/
|   |-- data/house_prices.csv
|   |-- house_price_model.ipynb
|   `-- pipeline.py
|-- models/
|   |-- house_price.pkl
|   |-- model_metadata.json
|   `-- locations.json
|-- backend/
|   |-- app/
|   |   |-- api/routes/
|   |   |-- core/
|   |   |-- schemas/
|   |   |-- services/
|   |   |   |-- prediction/
|   |   |   |-- cad_generation/
|   |   |   |-- cad_extraction/
|   |   |   `-- neighborhood/
|   |   `-- main.py
|   |-- tests/
|   |-- requirements.txt
|   |-- .env.example
|   `-- Dockerfile
|-- frontend/
|   |-- src/
|   |   |-- api/
|   |   |-- components/
|   |   |-- features/
|   |   |   |-- prediction/
|   |   |   |-- cad-editor/
|   |   |   |-- cad-extractor/
|   |   |   |-- properties/
|   |   |   `-- neighborhood/
|   |   |-- pages/
|   |   |-- stores/
|   |   |-- types/
|   |   `-- App.tsx
|   |-- public/
|   |-- .env.example
|   `-- package.json
|-- docs/
|   |-- screenshots/
|   `-- architecture.md
|-- PROJECT_PLAN.md
|-- README.md
`-- .gitignore
```

## 9. Implementation Phases

### Phase 0 - Repository and Contract Audit

- Confirm tool versions.
- Confirm Git remote and repository cleanliness.
- Verify the raw data schema.
- Test the existing parsing functions.
- Define the exact model feature contract shared by notebook, backend, and frontend.
- Establish focused validation commands.

Exit condition: one documented and tested feature schema with no ambiguity about training versus inference fields.

### Phase 1 - Notebook and Model

- Create `notebooks/house_price_model.ipynb`.
- Load and inspect the complete dataset.
- Add at least four meaningful EDA plots with written interpretations.
- Apply and explain cleaning and feature engineering.
- Prevent target leakage and duplicate leakage.
- Train at least two models.
- Evaluate on a held-out test set with MAE, RMSE, and R2.
- Add predicted-versus-actual visualization and comparison table.
- Select and justify the final model.
- Export the full model pipeline, metadata, and allowed locations.
- Restart the kernel and run every cell from top to bottom.

Exit condition: reproducible notebook plus a reloaded model that successfully predicts a sample.

### Phase 2 - Minimal Real Backend

- Scaffold FastAPI using the final model contract.
- Implement `/health`, `/model/metadata`, `/locations`, and `/predict`.
- Load the model once at startup.
- Add validation, CORS, structured errors, and tests.
- Verify predictions through Swagger and automated tests.

Exit condition: a real request returns a real model prediction with passing tests.

### Phase 3 - Minimal End-to-End Frontend

- Scaffold React + TypeScript + Vite.
- Establish the reference-inspired application shell.
- Build the property form and summary panel.
- Connect the form to FastAPI.
- Implement validation, loading, failure, and formatted-result states.
- Verify the full browser flow before adding advanced features.

Exit condition: the user can submit property data and see a real prediction in the browser.

### Phase 4 - CAD Editor Foundation

- Define the shared CAD scene schema.
- Implement SVG/Canvas rendering, selection, pan, zoom, and basic editing.
- Add walls, rooms, doors, windows, labels, and dimensions.
- Add undo/redo and serialization.
- Add SVG/JSON export before attempting AI generation.

Exit condition: a manually created plan can be edited, saved, reloaded, and exported.

### Phase 5 - AI CAD Drawer

- Convert form data and text instructions into layout constraints.
- Implement a procedural layout baseline.
- Validate geometry and circulation.
- Generate multiple variations where practical.
- Send the selected scene into the existing editor.
- Add an optional 3D extrusion preview.

Exit condition: structured property requirements produce an editable, geometrically valid plan.

### Phase 6 - AI CAD Extractor

- Implement upload and image preprocessing.
- Add wall/line, opening, room, symbol, and OCR detection.
- Vectorize detections into the shared scene schema.
- Add scale calibration and confidence overlays.
- Add a correction/review workflow.
- Evaluate on several clean, scanned, photographed, and hand-drawn plans.

Exit condition: a supported floor-plan image becomes a useful editable plan with transparent uncertainty.

### Phase 7 - Multiple Units and Neighborhood Map

- Add project/unit state and unit cards.
- Add duplication and comparison.
- Add map search, markers, dataset statistics, and price tiers.
- Add nearby geographic context through the chosen map provider.
- Connect map selection to the property form and prediction workflow.

Exit condition: users can compare several units and understand their location context.

### Phase 8 - Visual Polish and Accessibility

- Match the reference hierarchy, proportions, spacing, border treatments, and monochrome palette.
- Add subtle interaction motion without distracting from the workspace.
- Complete responsive layouts.
- Add keyboard support, focus states, labels, and accessible contrast.
- Add skeleton, empty, error, and recovery states throughout.

Exit condition: the product feels cohesive and professional across desktop, tablet, and mobile.

### Phase 9 - Verification, Documentation, and Deployment

- Run notebook top-to-bottom.
- Run backend tests.
- Run frontend typecheck, lint, tests, and production build.
- Verify the real browser flow and network responses.
- Test representative valid and invalid cases.
- Write the complete README required by the PDF.
- Add architecture documentation, API examples, metrics, and screenshots.
- Perform a clean-clone setup test using only the README.
- Deploy the frontend and backend.
- Verify public URLs and the public prediction flow.
- Confirm the remote Git commit matches the intended local revision.

Exit condition: a stranger can clone or open the deployed application and reproduce the demonstrated behavior.

## 10. Testing and Quality Gates

### Machine Learning

- Parser unit tests for Lac/Crore prices, sqft/sqm/sqyrd/acre areas, floor descriptions, and counts.
- Explicit target-leakage checks.
- Duplicate analysis before splitting.
- Reproducible metrics and model reload test.
- Sanity predictions for realistic low, median, and high-value properties.

### Backend

- Health endpoint.
- Valid prediction.
- Missing/invalid fields return 422.
- Unknown categories do not crash inference.
- Startup failure is clear when the model is missing or incompatible.
- CAD payload validation.
- Image upload limits and unsupported-file handling.

### Frontend

- Form validation.
- Loading and API-error states.
- Correct currency formatting.
- Add/remove/duplicate unit flow.
- CAD editor interactions and serialization.
- Image extraction review flow.
- Responsive and accessibility checks.

### End-to-End

- Browser form to real API to real model.
- Create multiple units and compare them.
- Generate a CAD plan and edit it.
- Extract a supported plan image and correct uncertain elements.
- Select a map location and receive dataset-derived neighborhood information.
- Verify public deployment without mocks.

## 11. Deployment Strategy

- Frontend: Vercel.
- Backend: a Python-capable service such as Render, Railway, Fly.io, or another suitable container host.
- Model artifact: bundled with the backend when below provider limits, otherwise downloaded from controlled storage during deployment.
- Map credentials: environment variables only.
- Local development must continue working without production credentials; map-dependent features should explain what is unavailable rather than crash.

## 12. Git Workflow

- Keep commits small, meaningful, and buildable.
- Commit after each completed and validated milestone rather than after every temporary edit.
- Never commit secrets, `.env`, `.venv`, `node_modules`, raw CSV data, temporary renders, or local caches.
- Scan staged changes for credentials before publishing.
- Push completed milestones to `YUST777/iti_ai_project`.
- Verify the remote branch after each push.

Suggested milestone commits:

1. `docs: add project architecture and delivery plan`
2. `feat: complete house price training notebook`
3. `feat: serve real predictions through FastAPI`
4. `feat: add reference-inspired prediction workspace`
5. `feat: add editable CAD scene editor`
6. `feat: generate CAD plans from property requirements`
7. `feat: extract editable plans from uploaded images`
8. `feat: add multi-unit and neighborhood comparison`
9. `docs: complete setup guide and deployment evidence`

## 13. Scope Priorities

### Required for Assignment Completion

1. Correct notebook and cleaning.
2. Two evaluated models and exported pipeline.
3. FastAPI prediction endpoint and tests.
4. React form and real result.
5. Complete README and clean repository.

### Product Differentiators

1. Reference-quality workspace UI.
2. Editable CAD editor.
3. AI CAD drawer.
4. AI CAD extractor.
5. Multiple-unit comparison.
6. Neighborhood map intelligence.
7. 3D preview.

The differentiators must not delay the verified ML submission foundation.

## 14. What We Start With

Start with **Phase 0 and Phase 1: lock the model feature contract and complete the notebook**.

The first concrete work session should:

1. Add parser and feature-contract tests around `notebooks/pipeline.py`.
2. Decide the exact inference fields that the frontend and backend will use.
3. Create `notebooks/house_price_model.ipynb`.
4. Produce the required EDA and cleaning narrative.
5. Train and compare the first two models.
6. Export a reloadable pipeline and verify one real prediction.

This is the correct starting point because every other feature depends on a stable definition of a property. The backend request schema, left-side UI fields, right-side summary, unit comparison, neighborhood statistics, and AI CAD constraints all need that shared property contract.

The first visible UI milestone comes immediately after the prediction API works: build the reference-inspired shell and connect the left property form to a real prediction displayed in the right summary panel. The central CAD canvas is then added on top of a functioning product rather than used to hide an unfinished model.

