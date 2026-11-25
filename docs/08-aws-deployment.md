# Phase 7 — AWS Deployment Guide

This guide walks through deploying the Multi-User RAG Platform on AWS, from a
blank EC2 instance to a fully working HTTPS deployment with S3-backed uploads.
It targets the **free-tier-friendly** path first and then explains how to scale
up later.

The pieces you already have:

| Component | Path |
| --- | --- |
| Backend image | `backend/Dockerfile` (multi-stage, non-root, embedding model cached) |
| Frontend image | `frontend/Dockerfile` (multi-stage, non-root) |
| Compose (dev) | `docker-compose.yml` |
| Compose (prod) | `docker-compose.prod.yml` |
| Nginx reverse proxy | `deploy/nginx.conf` |
| CI | `.github/workflows/ci.yml` |
| CD | `.github/workflows/deploy.yml` |

---

## 1. Architecture overview

```
                         ┌──────────────────────── EC2 (Ubuntu) ───────────────────────┐
                         │                                                              │
   Internet ──443──▶ Nginx │  /        -> Streamlit :8501                              │
   (HTTPS)                 │  /api     -> FastAPI   :8000                              │
                         │  /health  -> FastAPI   :8000                              │
                         │                                                              │
                         │   backend  ──▶ PostgreSQL 15 (container, volume)            │
                         │   backend  ──▶ ChromaDB     (container, volume)             │
                         │   frontend ──▶ (browser via Nginx)                          │
                         └──────────────────────────────────────────────────────────────┘
                                     │
                                     └──▶ S3 (document uploads, optional)
```

- **Nginx** is the single public entry point: it terminates TLS and routes
  `/` to Streamlit and `/api` + `/health` to FastAPI.
- **PostgreSQL** and **ChromaDB** run as containers with persisted volumes.
- **S3** stores uploaded document files when `STORAGE_BACKEND=s3`.

---

## 2. Instance recommendation

### Recommended: `t3.medium` (4 vCPU, 4 GB RAM)

**Why this size:**

- **RAM is the binding constraint.** The backend loads a ~130 MB sentence
  transformer model and the ChromaDB process keeps the index in memory. On a
  1 GB machine these two, plus PostgreSQL and Nginx, will hit swap and become
  unresponsive. 4 GB leaves comfortable headroom for the model, the vector
  index, and a few concurrent users.
- **4 vCPUs** keep PDF/text extraction (CPU-bound) responsive while the model
  runs inference for a query, and let Docker run all four containers without
  starvation.
- **`t3` burstable** is the cheapest general-purpose family; the platform's
  load is bursty (upload + embed on document add, then idle), which matches the
  burst-credit model well.
- **`t3.medium` is eligible for the AWS Free Tier** (750 hours/month for the
  first 12 months on a t2/t3 micro/small/medium), so a single instance can run
  essentially free during the trial period.

### Alternatives

| Need | Instance | Why |
| --- | --- | --- |
| Lowest cost / testing | `t3.small` (2 GB) | Works for single-user demos; tight when embedding many docs |
| Heavier load / many users | `t3.large` (8 GB) | More headroom for the model + larger vector stores |
| Always-on, predictable | `m6i.large` | Fixed performance, no burst-credit concerns; higher hourly cost |

> **EMR/ECS not needed yet.** A single EC2 instance running Compose is the
> simplest production shape for this workload. See §8 for when to move off it.

---

## 3. Launch the EC2 instance

1. In the AWS console go to **EC2 → Instances → Launch instances**.
2. **Name**: `rag-platform`.
3. **AMI**: *Ubuntu 22.04 LTS* (64-bit).
4. **Instance type**: `t3.medium`.
5. **Key pair**: create or select a key pair you can download (`rag-key.pem`).
   This is the only way to SSH in initially.
6. **Network settings**: default VPC, default subnet.
7. **Advanced → IAM role** (optional but recommended): attach an inline policy
   granting `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on your uploads
   bucket so you do not need to store S3 access keys on the box (see §6.3).
8. **Storage**: 30 GB gp3 is plenty (images + Postgres/Chroma volumes).
9. Launch.

### Assign an Elastic IP

A restart of the instance re-assigns its public IP. Attach a stable **Elastic
IP** so your DNS record never needs updating:

```bash
# Allocate (once)
aws ec2 allocate-address --domain ipv4
# Associate with the instance
aws ec2 associate-address --instance-id i-0123456789abcdef0 --allocation-id eipalloc-0123456789abcdef0
```

Note the **Elastic IP** and use it for the DNS A record in §7. Elastic IPs on
EC2 are free as long as the instance is running.

---

## 4. Security groups

Create a security group for the instance with these inbound rules:

| Type | Protocol | Port | Source | Purpose |
| --- | --- | --- | --- | --- |
| SSH | TCP | 22 | Your IP only (or a bastion) | Administrative access |
| HTTP | TCP | 80 | 0.0.0.0/0 | Let's Encrypt HTTP-01 + HTTP→HTTPS redirect |
| HTTPS | TCP | 443 | 0.0.0.0/0 | Public traffic |

Outbound: allow **all** (default) so the instance can reach ECR, S3, and
Let's Encrypt.

> Do **not** expose ports 8000/8501/5432/8001 to the internet. Only Nginx
> (80/443) is public; the Docker services are reached internally by
> Compose. Postgres and Chroma stay inside the Docker network.

---

## 5. Install Docker on the instance

On the EC2 instance (as `ubuntu`):

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# Add Docker's official GPG key and repo
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow the ubuntu user to use Docker
sudo usermod -aG docker ubuntu
sudo systemctl enable --now docker

# Log out and back in (or run:)
newgrp docker
```

Verify:

```bash
docker --version
docker compose version
```

### Install Nginx

```bash
sudo apt install -y nginx
sudo systemctl enable --now nginx
```

---

## 6. Configure environment variables

All configuration comes from environment variables — **no secrets are ever
hardcoded**. Create the production override `docker-compose.prod.yml` values via
a `.env` file or SSM Parameter Store.

### 6.1 Create the deployment directory

```bash
sudo mkdir -p /opt/rag
sudo chown ubuntu:ubuntu /opt/rag
```

Copy the repository files (or clone) into `/opt/rag`, then create
`/opt/rag/.env.prod`:

```bash
cat > /opt/rag/.env.prod <<'EOF'
# ---- Core ----
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://rag:CHANGE_ME@postgres:5432/rag
JWT_SECRET=REPLACE_WITH_A_LONG_RANDOM_STRING_MIN_32_CHARS
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=https://your-domain.com

# ---- Storage: S3 for uploads ----
STORAGE_BACKEND=s3
S3_REGION=us-east-1
S3_ENDPOINT_URL=
S3_BUCKET=rag-uploads-<your-account>
S3_PREFIX=uploads
# Prefer an IAM role over keys (see §6.3). If you must use keys, set these:
# S3_ACCESS_KEY=
# S3_SECRET_KEY=

# ---- Retrieval tuning ----
RETRIEVAL_TOP_K=5
RETRIEVAL_MIN_SCORE=0.2
RERANK_ENABLED=false
EOF
```

The backend `docker-compose.yml` service reads `env_file: backend/.env`, but
for production the `docker-compose.prod.yml` override supplies the real values
from this file (set `env_file` in the override to `.env.prod`, or export the
vars before `docker compose`).

> `JWT_SECRET` must be at least 32 random characters and **overridden** in
> production — it has a local-dev default only to make `docker compose up`
> work out of the box.

### 6.2 Option A — SSM Parameter Store (recommended)

Instead of a plaintext `.env`, store secrets in AWS Systems Manager Parameter
Store and let the instance fetch them:

```bash
aws ssm put-parameter --name "/rag/prod/JWT_SECRET" \
  --value "$(openssl rand -base64 48)" --type SecureString
aws ssm put-parameter --name "/rag/prod/S3_BUCKET" \
  --value "rag-uploads-<your-account>" --type String
# ...one parameter per secret...
```

On the instance, install the SSM Agent (present on Amazon Linux; on Ubuntu add
the CloudWatch agent) and read them at deploy time. A common pattern is a small
deploy script that renders `/opt/rag/.env.prod` from SecureString parameters:

```bash
sudo -u ubuntu bash -c '
  {
    echo "ENVIRONMENT=production"
    echo "DATABASE_URL=postgresql+asyncpg://rag:CHANGE_ME@postgres:5432/rag"
    echo "JWT_SECRET=$(aws ssm get-parameter --name /rag/prod/JWT_SECRET \
        --query Parameter.Value --output text)"
    echo "STORAGE_BACKEND=s3"
    echo "S3_BUCKET=$(aws ssm get-parameter --name /rag/prod/S3_BUCKET \
        --query Parameter.Value --output text)"
  } > /opt/rag/.env.prod'
```

SecureString parameters are encrypted at rest with KMS and never appear in the
Compose file or shell history.

### 6.3 Option B — IAM role for S3 (no S3 keys at all)

The cleanest approach is to grant the instance's IAM role access to the bucket
and leave `S3_ACCESS_KEY` / `S3_SECRET_KEY` empty. boto3 automatically uses the
instance profile credentials:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::rag-uploads-<your-account>/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::rag-uploads-<your-account>"
    }
  ]
}
```

`S3StorageBackend` uses `boto3.client("s3", region_name=...)` which picks up the
instance role automatically.

### 6.4 Create the S3 bucket

```bash
aws s3api create-bucket --bucket rag-uploads-<your-account> --region us-east-1
# Enable versioning (optional, protects against accidental overwrite)
aws s3api put-bucket-versioning --bucket rag-uploads-<your-account> \
  --versioning-configuration Status=Enabled
```

The `S3_PREFIX` (default `uploads`) namespaces objects as
`uploads/<user_id>/<document_id>/<filename>`, so multiple users and documents
never collide.

---

## 7. Domain + HTTPS (Let's Encrypt)

### 7.1 Point the domain at the Elastic IP

In your DNS provider, add an **A record**:

```
your-domain.com  ->  <your-elastic-ip>
```

Wait for propagation (a few minutes to a few hours). Verify:

```bash
dig +short your-domain.com
# should print the Elastic IP
```

### 7.2 Configure Nginx

Edit `deploy/nginx.conf` and replace the two `rag.example.com` placeholders and
certificate paths with your domain:

```
server_name your-domain.com;
ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
```

Install and enable it:

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/rag
sudo ln -sf /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/rag
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### 7.3 Obtain the certificate with certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo mkdir -p /var/www/certbot
sudo certbot --nginx -d your-domain.com --non-interactive --agree-tos \
  --email you@example.com --redirect
```

certbot edits the Nginx config to serve HTTPS and set up the 301 redirect
automatically, and installs a systemd timer for automatic renewal. Test renewal:

```bash
sudo certbot renew --dry-run
```

### 7.4 Verify end-to-end

```bash
curl -fsS https://your-domain.com/health/live
# {"status":"ok",...}
curl -fsS https://your-domain.com/_stcore/health
# ok
```

Open `https://your-domain.com` in a browser — you should see the Streamlit
login page.

---

## 8. Run the stack

On the instance in `/opt/rag`:

```bash
# Apply the production override (set IMAGE_* if you built/pushed to ECR)
export IMAGE_BACKEND=<ecr>/backend:latest
export IMAGE_FRONTEND=<ecr>/frontend:latest

docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

The backend runs `alembic upgrade head` on startup before serving, so the
database schema is always up to date with the image.

**First-run user:** register through the UI. The default role is `user`; create
or promote an admin via the database if you need `admin` privileges:

```bash
docker compose exec postgres psql -U rag -d rag -c \
  "UPDATE users SET role='admin' WHERE email='you@example.com';"
```

### Updates

Re-run the CI/CD deploy workflow (tag a new version, or run
*Deploy → workflow_dispatch* and pass the `health_url` input, e.g.
`https://your-domain.com/health/live`) or, manually:

```bash
cd /opt/rag
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 9. Cost estimate (free-tier friendly)

Monthly estimates for the recommended single-instance setup. AWS prices vary
by region and change over time — treat these as ballpark figures and check the
pricing calculator for your region.

| Component | Resource | Approx. monthly cost | Notes |
| --- | --- | --- | --- |
| Compute | EC2 `t3.medium` (750 h) | **$0** (Free Tier, first 12 mo) then ~$21 | 4 vCPU / 4 GB |
| Compute | EC2 `t3.small` (fallback) | **$0** (Free Tier) then ~$14 | 2 GB; tighter |
| Storage | 30 GB gp3 EBS | ~$2.40 | Root volume for images + logs |
| Network | Elastic IP | **$0** | Free while instance runs |
| Network | Data transfer (egress) | ~$0.09/GB | Small for typical RAG use |
| Database | PostgreSQL **on EC2** (container) | **$0** extra | Uses the instance; no RDS yet |
| Vector store | ChromaDB **on EC2** (container) | **$0** extra | Uses the instance |
| Object storage | S3 (~5 GB) | ~$0.12 | First 50 GB free tier |
| DNS | Route 53 hosted zone | ~$0.50 | Per domain, if using Route 53 |
| Certificates | Let's Encrypt | **$0** | Auto-renewed |
| **Total (Free Tier)** | | **~$3–$5 / month** | Dominated by DNS + S3 + storage |
| **Total (after Free Tier)** | | **~$25–$40 / month** | Single `t3.medium` + EBS + S3 |

Cost levers:

- **Keep it on one instance** for as long as possible — that is the whole
  point of this architecture. Postgres and Chroma ride along for free.
- **Turn the instance off** when not in use (dev/test) — you only pay for
  compute hours.
- **Use S3 lifecycle rules** to expire old uploads and keep the bucket tiny.

---

## 10. Scaling later

The single-EC2 design is deliberately simple and handles a lot. When you outgrow
it, here is the staged path. Do **not** jump to the bottom of this list
prematurely — each step adds cost and operational complexity.

### 10.1 RDS PostgreSQL (move off the containerized DB)

**When:** you need managed backups, automated failover, read replicas, or the
database is the bottleneck under many concurrent writers.

**How:** replace the `postgres` service in `docker-compose.prod.yml` with an
RDS instance (e.g. `db.t4g.micro` in the free tier, then `db.t4g.small`).
Point `DATABASE_URL` at the RDS endpoint and remove the `pgdata` volume.
Keep a security group that allows the backend container/EC2 to reach RDS on
5432 and no public access.

### 10.2 ElastiCache (Redis) for rate limiting and caching

**When:** you need shared rate-limit state across multiple backend instances,
session caching, or to cut latency on repeated queries.

**How:** the rate limiter is already behind the `RateLimiter` abstraction, so
swapping the in-memory implementation for a Redis-backed one does not change
any call site. Provision a `cache.t4g.micro`/`small` ElastiCache Redis node,
point it at the backend via a security group, and implement
`RedisRateLimiter` behind the same interface.

### 10.3 ECS / Fargate (move off EC2 for compute)

**When:** you need to run multiple backend instances behind a load balancer for
HA, or your team wants container orchestration without managing the EC2 host.

**How:**
1. Push images to ECR (the CI/CD workflow already does this).
2. Define an ECS Fargate task for `backend` and one for `frontend`.
3. Use an **Application Load Balancer** in front of the frontend task and an
   **internal ALB** in front of the backend task (replacing the direct Nginx
   `/api` route).
4. Replace containerized Postgres/Chroma with RDS + a managed vector store
   (see 10.4), since Fargate tasks are ephemeral.
5. Store secrets in SSM Parameter Store / Secrets Manager and reference them
   in the task definition.

### 10.4 Managed vector store

**When:** ChromaDB on EC2/Fargate becomes a scaling or HA liability as the
document set grows.

**Options:**
- **Amazon Bedrock Knowledge Bases** — serverless vector store + retrieval,
  integrates with the existing retrieval pipeline via the `VectorStore`
  interface.
- **Amazon MemoryDB / OpenSearch** — for larger, always-on vector workloads.

Because the `VectorStore` abstraction isolates the backend from the concrete
store, adding a new implementation does not touch the rest of the code.

### 10.5 Auto Scaling

**When:** traffic is spiky and you want to survive bursts.

On EC2: an Application Auto Scaling policy on the ASG behind your load
balancer. On ECS/Fargate: target-tracking CPU/memory scaling on the service.
Start with a modest max (2–3 tasks) to control cost.

### Summary table

| Stage | Adds | Cost impact | Trigger |
| --- | --- | --- | --- |
| 1. RDS PostgreSQL | Managed DB, backups, HA | +~$15–$25/mo | DB is the bottleneck |
| 2. ElastiCache Redis | Shared rate-limit/cache | +~$12–$20/mo | Multiple backends, caching |
| 3. ECS Fargate | Container orchestration, HA compute | +~$30–$70/mo | Need >1 backend, managed hosts |
| 4. Managed vector store | Bedrock KB / OpenSearch | + varies | Large/HA vector workload |
| 5. Auto Scaling | Elastic capacity | Pay per use | Traffic bursts |

---

## 11. Pre-flight checklist

- [ ] EC2 `t3.medium` launched, Elastic IP attached.
- [ ] Security group: 22 (restricted), 80, 443 open; DB ports internal.
- [ ] Docker + Nginx installed, `ubuntu` in the `docker` group.
- [ ] `/opt/rag` populated with repo files.
- [ ] `.env.prod` created (or SSM Parameter Store configured).
- [ ] `JWT_SECRET` set to a long random value (≥ 32 chars).
- [ ] S3 bucket created and reachable (IAM role or keys).
- [ ] `STORAGE_BACKEND=s3` set for production uploads.
- [ ] Domain A record → Elastic IP.
- [ ] Nginx installed from `deploy/nginx.conf`, certbot HTTPS + redirect live.
- [ ] `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`
      healthy; `/health/live` returns 200.
- [ ] First user registered; admin role granted if needed.
