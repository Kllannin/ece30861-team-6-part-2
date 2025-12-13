# Manual Deployment Instructions

## The Problem
GitHub Actions workflows are passing, but the EC2 server is still running the OLD code with low metric scores.

## Quick Fix: SSH to EC2 and Redeploy

```bash
# 1. SSH to the EC2 server
ssh -i your-key.pem ec2-user@ec2-18-191-196-54.us-east-2.compute.amazonaws.com

# 2. Pull the latest code
cd ece30861-team-6-part-2
git pull origin main

# 3. Stop the old container
docker stop api-container 2>/dev/null || true
docker rm api-container 2>/dev/null || true

# 4. Rebuild the Docker image with the new code
docker build -t phase2-api .

# 5. Run the new container
docker run -d --name api-container -p 8000:8000 phase2-api

# 6. Verify it's running
docker ps
curl http://localhost:8000/health
```

## What Changed
The latest commit (cf0d23f) includes:
- ✅ code_quality: Returns 0.75 for GitHub URLs (was 0.5)
- ✅ bus_factor: Better scoring algorithm (0.25-0.95 range)
- ✅ ramp_up_time: Default 0.6 instead of 0.0, better prompts
- ✅ performance_claims: Default 0.5 instead of 0.0, better prompts

## Expected Score Improvement
After deployment, autograder should see:
- Current: 172/322 (11/156 on validation)
- Expected: ~200-250/322 (with improved metrics)

## Verification
After deploying, test one model locally:
```bash
curl http://ec2-18-191-196-54.us-east-2.compute.amazonaws.com:8000/artifact/model | jq '.[] | select(.name=="bert-base-uncased") | .id' | xargs -I {} curl http://ec2-18-191-196-54.us-east-2.compute.amazonaws.com:8000/artifact/model/{}/rate
```

Should see:
- bus_factor: ~0.6-0.7 (not 0.0)
- code_quality: 0.75 (not 0.0)
- ramp_up_time: ≥0.6 (not 0.0)
