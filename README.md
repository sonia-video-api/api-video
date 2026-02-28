# MonAPI Video — Génération de Vidéos IA

API de génération de vidéos IA déployée sur Render, utilisant Fal.ai ou Replicate comme moteur GPU externe.

## Architecture

```
Render (API FastAPI) → Fal.ai ou Replicate (GPU) → Vidéo MP4
```

## Tarifs

| Durée | Fal.ai | Replicate |
|-------|--------|-----------|
| 5 secondes | ~0.25$ | ~0.90$ |
| 10 secondes | ~0.50$ | ~1.30$ |

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Page d'accueil |
| POST | `/admin/create-key` | Créer une clé API |
| POST | `/v1/video` | Générer une vidéo |
| GET | `/v1/jobs/{id}` | Voir le statut d'un job |
| GET | `/v1/jobs` | Lister tous les jobs |
| GET | `/docs` | Documentation Swagger |

## Variables d'environnement

| Variable | Description |
|----------|-------------|
| `FAL_API_KEY` | Clé API Fal.ai (recommandé) |
| `REPLICATE_API_TOKEN` | Token Replicate (alternatif) |

## Utilisation

```bash
# 1. Créer une clé API
curl -X POST https://monapi-video-online.onrender.com/admin/create-key

# 2. Générer une vidéo
curl -X POST https://monapi-video-online.onrender.com/v1/video \
  -H "Authorization: Bearer mak_xxx" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "sunset over ocean, cinematic", "seconds": 6}'

# 3. Vérifier le statut
curl https://monapi-video-online.onrender.com/v1/jobs/job_XXXXX \
  -H "Authorization: Bearer mak_xxx"
```
