
# 🧠 1. Requirements

### ✅ Functional Requirements

* Convert long URL → short URL
* Redirect short URL → original URL
* Custom alias (optional)
* Track clicks (analytics)

### ⚡ Non-Functional Requirements

* High availability (always redirect fast)
* Low latency (redirect in milliseconds)
* Scalable (millions of URLs/day)
* Reliable (no data loss)

---

# 🏗️ 2. High-Level Design

**Core flow:**

```
User → API Server → Database → Short URL
User → Short URL → Redirect Service → Original URL
```

### Components:

* API Server (create short URL)
* Redirect Service (handle redirection)
* Database (store mappings)
* Cache (fast lookup)
* Load Balancer

---

# 🔑 3. URL Shortening Logic

We need to generate a **unique short key**

### Approach 1: Base62 Encoding (Most common)

* Use characters: `a-zA-Z0-9` → 62 chars
* Convert ID → short string

Example:

```
ID: 125 → Encoded: "cb"
```

👉 Why Base62?

* Shorter URLs
* URL-safe
* Scales well

---

# 🗄️ 4. Database Design

### Table: `url_mapping`

| Field       | Type        |
| ----------- | ----------- |
| id          | BIGINT (PK) |
| short_code  | VARCHAR     |
| long_url    | TEXT        |
| created_at  | TIMESTAMP   |
| expiry_date | TIMESTAMP   |
| user_id     | OPTIONAL    |

---

# ⚡ 5. API Design

### Create Short URL

```
POST /shorten
Body: { "url": "https://example.com" }

Response:
{ "short_url": "bit.ly/abc123" }
```

### Redirect

```
GET /abc123 → 302 Redirect → original URL
```

---

# 🚀 6. Scaling the System

### 🔹 Problem: High Read Traffic (redirects)

Solution:

* Use **Cache (Redis)**

  * Key: short_code
  * Value: long_url

👉 90% requests served from cache → super fast

---

### 🔹 Problem: Database bottleneck

Solution:

* Read replicas
* Sharding (based on short_code hash)

---

### 🔹 Problem: ID generation

Options:

* Auto-increment (simple but single DB bottleneck)
* Distributed ID generators (like Snowflake)

---

# ⚡ 7. Caching Strategy

* Cache popular URLs
* Use **LRU eviction**
* TTL for expiration

Flow:

```
Request → Check Cache → If miss → DB → Update Cache
```

---

# 🔁 8. Redirection Flow

```
User clicks short URL
→ Load Balancer
→ Redirect Service
→ Cache lookup
→ DB (if cache miss)
→ 302 Redirect
```

---

# 📊 9. Analytics (Bonus Feature)

Track:

* Click count
* Location
* Device type

How:

* Use async processing (Kafka / queue)
* Store in analytics DB

---

# 🛡️ 10. Edge Cases

* Duplicate URLs → same short URL? (optional optimization)
* Expired links
* Malicious URLs (add validation)
* Custom aliases conflict

---

# ⚖️ 11. Trade-offs

| Approach      | Pros             | Cons                |
| ------------- | ---------------- | ------------------- |
| Base62        | Short, efficient | Needs ID generation |
| Hashing (MD5) | No DB dependency | Collisions possible |

---

# 🧩 Final Architecture (Simplified)

```
          [User]
             ↓
      [Load Balancer]
             ↓
   ┌───────────────┐
   │ API Servers   │
   └───────────────┘
       ↓       ↓
   [Cache]   [DB]
       ↓
 [Redirect Service]
```

---

# 🔥 How to Impress Interviewer

If you say these → you're ahead of 90%:

* “I’ll optimize for read-heavy traffic using Redis cache”
* “Use Base62 encoding for compact URLs”
* “Introduce async analytics pipeline”
* “Handle hot URLs using CDN/cache”

# 🧠 1. Requirements

### ✅ Functional

* User can **create a paste** (text snippet)
* Generate a **unique URL**
* Retrieve paste via URL
* Optional:

  * Expiry time (self-destruct)
  * Public / private paste
  * Custom alias

### ⚡ Non-Functional

* High availability
* Low latency (fast reads)
* Scalable (millions of pastes)
* Durable (no data loss)

---

# 🏗️ 2. High-Level Design

**Flow:**

```
User → API → Store Paste → Generate URL
User → Short URL → Fetch Paste
```

### Components:

* API servers
* Database (store text)
* Cache (fast reads)
* Load balancer
* Object storage (optional for large pastes)

---

# 🔑 3. Paste Creation Logic

We need a **unique paste ID**

### Approach: Base62 Encoding

* Same idea as URL shortener
* Generate ID → encode → short URL

Example:

```
ID: 98765 → "AbX9"
URL: paste.app/AbX9
```

---

# 🗄️ 4. Database Design

### Table: `pastes`

| Field       | Type        |
| ----------- | ----------- |
| id          | BIGINT (PK) |
| short_code  | VARCHAR     |
| content     | TEXT        |
| created_at  | TIMESTAMP   |
| expiry_time | TIMESTAMP   |
| visibility  | ENUM        |

---

# ⚡ 5. API Design

### Create Paste

```
POST /paste
Body:
{
  "content": "Hello World",
  "expiry": "1 day",
  "visibility": "public"
}
```

Response:

```
{ "url": "paste.app/AbX9" }
```

---

### Get Paste

```
GET /AbX9
→ returns content
```

---

# 🚀 6. Scaling the System

### 🔹 Problem: Large text storage

Solution:

* Small pastes → DB (TEXT)
* Large pastes → Object storage (like S3)

---

### 🔹 Problem: Heavy read traffic

Solution:

* Use **Redis cache**
* Cache hot pastes

---

### 🔹 Problem: High write volume

Solution:

* Partition DB (sharding by ID)
* Use write-optimized storage

---

# ⚡ 7. Caching Strategy

```
Request → Cache → DB (if miss)
```

* Cache frequently accessed pastes
* TTL based on expiry

---

# 🔁 8. Expiry Handling

Two approaches:

### 1. Lazy Deletion (simple)

* Check expiry during read
* Ignore expired pastes

### 2. Background Job (better)

* Cron job deletes expired data

---

# 🔒 9. Security Considerations

* Prevent malicious scripts (XSS filtering)
* Rate limiting (avoid spam)
* Private pastes → add authentication/token

---

# 📊 10. Optional Features (Interview Bonus)

* Syntax highlighting (code pastes)
* Version history
* Paste editing
* Analytics (views count)

---

# ⚖️ 11. Trade-offs

| Design Choice    | Pros       | Cons                        |
| ---------------- | ---------- | --------------------------- |
| Store in DB      | Simple     | Not scalable for large text |
| Object storage   | Scalable   | Slight latency              |
| Cache everything | Fast reads | Memory cost                 |

---

# 🧩 Final Architecture

```
         [User]
            ↓
    [Load Balancer]
            ↓
      [API Servers]
        ↓      ↓
   [Cache]   [DB]
                ↓
        [Object Storage]
```

# 🧠 1. Requirements

### ✅ Functional

* Limit number of requests per user/IP/API key
* Example: **100 requests per minute**
* Reject or throttle excess requests

### ⚡ Non-Functional

* Very low latency (runs on every request)
* Highly scalable (millions of users)
* Accurate (avoid unfair blocking)
* Fault-tolerant

---

# 🏗️ 2. High-Level Design

```
Client → API Gateway → Rate Limiter → API Server
```

### Components:

* API Gateway / Middleware (where limiter sits)
* Rate limiter service
* Fast storage (Redis)
* Config store (limits per user/plan)

---

# 🔑 3. Core Algorithms (VERY IMPORTANT)

## 1. Fixed Window Counter

* Count requests in fixed window (e.g., 1 min)

```
User A → 100 req in 00:00–00:59 → Allowed
User A → 100 req in 01:00–01:59 → Allowed
```

❌ Problem: Burst at boundary (can do 200 quickly)

---

## 2. Sliding Window Log

* Store timestamps of each request

✔ Accurate
❌ Memory heavy

---

## 3. Sliding Window Counter (Optimized)

* Combines fixed windows

✔ Balanced approach
✔ Used in real systems

---

## 4. Token Bucket (Most Popular 🚀)

* Tokens refill at fixed rate
* Each request consumes 1 token

```
Bucket size = 100
Refill = 10 tokens/sec
```

✔ Allows bursts
✔ Smooth traffic

---

## 5. Leaky Bucket

* Requests processed at constant rate

✔ Smooth output
❌ Can delay requests

---

# 🏆 Recommendation (Interview Answer)

Say this:

> “I’ll use **Token Bucket** because it supports burst traffic and smooth rate limiting.”

---

# ⚙️ 4. Data Storage Design

Use **Redis** (fast, atomic ops)

### Example Key:

```
rate_limit:user_123
```

### Value:

```
tokens: 45
last_refill: timestamp
```

---

# ⚡ 5. API Flow

```
Request comes in
→ Check Redis
→ If tokens available → allow
→ Else → reject (429 Too Many Requests)
```

---

# 🧮 6. Token Bucket Logic

On each request:

1. Calculate tokens to refill:

```
new_tokens = (current_time - last_refill) * rate
```

2. Update bucket
3. If tokens > 0 → allow
4. Else → reject

---

# 🚀 7. Scaling the System

### 🔹 Problem: Millions of users

Solution:

* Use **distributed Redis cluster**
* Hash users → different nodes

---

### 🔹 Problem: Single point of failure

Solution:

* Replication + failover
* Fallback (graceful degradation)

---

# ⚡ 8. Where to Implement?

### Option 1: API Gateway (Best)

* Centralized control
* Used in systems like Kong

### Option 2: Application Layer

* Flexible but duplicated logic

---

# 🔒 9. Advanced Features

* Different limits per user tier (free vs premium)
* IP-based + user-based limiting
* Distributed rate limiting
* Retry-after headers

---

# ⚖️ 10. Trade-offs

| Approach     | Pros         | Cons              |
| ------------ | ------------ | ----------------- |
| Fixed Window | Simple       | Burst issue       |
| Sliding Log  | Accurate     | High memory       |
| Token Bucket | Best balance | Slight complexity |

---

# 🧩 Final Architecture

```
        [Client]
            ↓
     [API Gateway]
            ↓
     [Rate Limiter]
            ↓
        [Redis]
            ↓
      [API Server]
```

# 🧠 1. Requirements

### ✅ Functional

* Upload files (images, videos, docs)
* Download/view files
* Share files (via link or user access)
* Folder structure
* Version history (optional)

### ⚡ Non-Functional

* High durability (no data loss 🚨)
* High availability
* Scalable (billions of files)
* Low latency (fast uploads/downloads)

---

# 🏗️ 2. High-Level Architecture

```
Client → API Gateway → Metadata Service → Storage Service
                          ↓
                       Database
                          ↓
                    Object Storage
                          ↓
                         CDN
```

### Components:

* API Gateway
* Auth Service
* Metadata Service
* File Storage Service
* Database (metadata)
* Object Storage (actual files)
* CDN (fast delivery)

---

# 📂 3. Key Idea (Very Important)

👉 **Never store files directly in DB**

Instead:

* DB → stores metadata
* Object storage → stores actual files

---

# 🗄️ 4. Database Design (Metadata)

### Table: `files`

| Field      | Type      |
| ---------- | --------- |
| file_id    | UUID      |
| user_id    | UUID      |
| file_name  | VARCHAR   |
| file_path  | TEXT      |
| size       | BIGINT    |
| created_at | TIMESTAMP |
| version    | INT       |
| is_deleted | BOOLEAN   |

---

# 📦 5. File Storage Design

Use object storage like:

* AWS S3 (conceptually)
* Distributed file systems

### Storage Strategy:

```
file_id → hashed → stored in buckets
```

Example:

```
file_id = abcd1234
→ hash → /ab/cd/abcd1234
```

✔ Prevents hotspots
✔ Scales well

---

# ⬆️ 6. Upload Flow

```
1. User uploads file
2. API authenticates user
3. Generate upload URL (pre-signed)
4. Client uploads directly to storage
5. Metadata saved in DB
```

👉 Why direct upload?

* Reduces server load
* Faster uploads

---

# ⬇️ 7. Download Flow

```
1. User requests file
2. Check permissions
3. Generate download URL
4. Serve via CDN
```

---

# ⚡ 8. Scaling Strategy

### 🔹 Problem: Large files (GBs)

Solution:

* Chunk upload (split into parts)
* Resume upload

---

### 🔹 Problem: High traffic

Solution:

* CDN (serve cached files globally)
* Cache metadata

---

### 🔹 Problem: Storage growth

Solution:

* Distributed object storage
* Data partitioning

---

# 🔄 9. File Versioning

Instead of overwrite:

* Store new version as new object
* Update metadata

Example:

```
file_v1 → file_v2 → file_v3
```

---

# 🔒 10. Security

* Authentication (JWT / OAuth)
* Access control (owner, shared users)
* Encryption:

  * At rest
  * In transit (HTTPS)

---

# 📊 11. Sharing System

### Two types:

1. Private sharing (user-based)
2. Public link (token-based)

Example:

```
drive.app/file/abc123?token=xyz
```

---

# ⚡ 12. Caching

* Cache metadata (Redis)
* Cache popular files via CDN

---

# 🧩 13. Final Architecture

```
        [Client]
            ↓
     [API Gateway]
            ↓
     [Auth Service]
            ↓
   [Metadata Service] → [DB]
            ↓
    [File Service]
            ↓
     [Object Storage]
            ↓
          [CDN]
```

---

# ⚖️ 14. Trade-offs

| Decision       | Pros              | Cons              |
| -------------- | ----------------- | ----------------- |
| Object storage | Scalable, durable | Slight latency    |
| Direct upload  | Efficient         | More complex flow |
| CDN            | Fast downloads    | Cost              |

# 🧠 1. Requirements

### ✅ Functional

* Send notifications via:

  * Email
  * SMS
  * Push (mobile/web)
* Support:

  * Immediate notifications
  * Scheduled notifications
  * Bulk notifications

---

### ⚡ Non-Functional

* Highly reliable (no message loss 🚨)
* Scalable (millions of users)
* Low latency (for real-time alerts)
* Retry mechanism for failures

---

# 🏗️ 2. High-Level Architecture

```id="q2y9sq"
Service → Notification Service → Queue → Workers → Providers
```

### Components:

* Notification API
* Message Queue
* Worker services
* Channel providers (Email/SMS/Push)
* Database (logs, preferences)

---

# 🔑 3. Key Design Principle

👉 **Decouple sending using a Queue (VERY IMPORTANT)**

Why?

* Avoid blocking main app
* Handle spikes smoothly
* Enable retries

---

# ⚙️ 4. Core Flow

```id="fd2hnn"
1. App triggers notification
2. Notification service validates request
3. Push message to queue
4. Worker picks message
5. Send via provider
6. Store status (success/failure)
```

---

# 📦 5. Message Queue

Use:

* Kafka / RabbitMQ / SQS (conceptually)

Message format:

```id="8yj95w"
{
  user_id: 123,
  type: "email",
  message: "Your order shipped",
  priority: "high"
}
```

---

# 👷 6. Worker Design

Workers:

* Consume from queue
* Route to correct channel

Example:

```id="e89htk"
if type == email → Email service
if type == sms → SMS gateway
if type == push → Push service
```

---

# 📧 7. Channel Services

### Email

* SMTP providers (like SendGrid)

### SMS

* SMS gateways (like Twilio)

### Push

* Firebase Cloud Messaging (FCM)
* Apple Push Notification Service (APNS)

---

# ⚡ 8. Scaling the System

### 🔹 Problem: High traffic spikes

Solution:

* Queue buffers traffic
* Auto-scale workers

---

### 🔹 Problem: Slow providers

Solution:

* Retry with backoff
* Fallback providers

---

# 🔁 9. Retry Mechanism

If sending fails:

```id="7nvjjr"
Retry after 1 min → 5 min → 15 min
```

* Use exponential backoff
* Move to **Dead Letter Queue (DLQ)** if permanent failure

---

# 🗄️ 10. Database Design

### Table: `notifications`

| Field      | Type      |
| ---------- | --------- |
| id         | UUID      |
| user_id    | UUID      |
| type       | ENUM      |
| status     | ENUM      |
| created_at | TIMESTAMP |

---

### Table: `user_preferences`

| Field     | Type    |
| --------- | ------- |
| user_id   | UUID    |
| email_opt | BOOLEAN |
| sms_opt   | BOOLEAN |
| push_opt  | BOOLEAN |

---

# 🎯 11. Personalization & Preferences

Before sending:

* Check user preferences
* Respect do-not-disturb settings

---

# ⚡ 12. Priority Handling

* High priority → immediate queue
* Low priority → batch processing

---

# 🔒 13. Security & Compliance

* Avoid spam
* Rate limiting per user
* GDPR / opt-out support

---

# 🧩 14. Final Architecture

```id="0kh9eo"
        [App Service]
              ↓
   [Notification Service]
              ↓
         [Queue]
              ↓
   ┌───────────────┐
   │   Workers     │
   └───────────────┘
     ↓      ↓      ↓
 [Email]  [SMS]  [Push]
```

---

# ⚖️ 15. Trade-offs

| Choice         | Pros               | Cons         |
| -------------- | ------------------ | ------------ |
| Queue-based    | Scalable, reliable | Slight delay |
| Direct sending | Fast               | Not scalable |

---

# 🔥 Interview Power Lines

Say these and you’ll stand out:

* “I’ll decouple using message queues for reliability”
* “Use retry with exponential backoff + DLQ”
* “Respect user notification preferences”
* “Design for idempotency to avoid duplicate sends”

# 🧠 1. Requirements

### ✅ Functional

* 1-to-1 messaging
* Group messaging
* Online/offline status
* Message delivery status (sent, delivered, read)
* Media sharing (images, videos)

---

### ⚡ Non-Functional

* Real-time delivery (low latency ⚡)
* High availability
* Scalable (billions of messages/day)
* Reliable (no message loss 🚨)

---

# 🏗️ 2. High-Level Architecture

```
Client → Load Balancer → Chat Servers → Message Queue → Storage
```

### Components:

* Chat servers (handle connections)
* Message queue
* Database (message storage)
* Presence service
* Notification service

---

# 🔑 3. Communication Protocol

👉 Use **WebSockets (persistent connection)**

Why not HTTP?

* HTTP = request/response
* Chat = real-time, bidirectional

---

# ⚙️ 4. Core Message Flow

```
1. User A sends message
2. Chat server receives it
3. Push to message queue
4. Deliver to User B (if online)
5. Store in DB
```

---

# 🧩 5. Real-Time Delivery

### Case 1: User B is ONLINE

```
Server → WebSocket → Instant delivery
```

### Case 2: User B is OFFLINE

```
Store message → Deliver when user reconnects
```

---

# 🗄️ 6. Database Design

### Table: `messages`

| Field       | Type      |
| ----------- | --------- |
| message_id  | UUID      |
| sender_id   | UUID      |
| receiver_id | UUID      |
| content     | TEXT      |
| timestamp   | TIMESTAMP |
| status      | ENUM      |

---

### Table: `conversations`

| Field        | Type  |
| ------------ | ----- |
| convo_id     | UUID  |
| participants | ARRAY |

---

# ⚡ 7. Scaling Strategy

### 🔹 Problem: Millions of concurrent connections

Solution:

* Use **stateless chat servers**
* Horizontal scaling
* Sticky sessions (same user → same server)

---

### 🔹 Problem: Message throughput

Solution:

* Use message queues (Kafka/RabbitMQ)
* Partition by user/conversation ID

---

# 🔄 8. Message Ordering

👉 Important!

Solution:

* Use timestamps + sequence IDs
* Partition messages per conversation

---

# ⚡ 9. Presence System (Online/Offline)

* Maintain in-memory store (Redis)

```
user_123 → online
```

* Update on connect/disconnect

---

# 📩 10. Push Notifications

If user offline:

* Send push via Firebase (FCM)
* Trigger notification system

---

# 📦 11. Media Handling

👉 Don’t send media via chat server

Flow:

```
1. Upload to storage (S3)
2. Send URL in message
```

---

# 🔒 12. Security

* End-to-end encryption (like WhatsApp)
* TLS for transport
* Authentication (JWT)

---

# 🔁 13. Reliability

* Message acknowledgments:

  * Sent
  * Delivered
  * Read

* Retry if not delivered

---

# 🧩 14. Final Architecture

```
        [Client]
            ↓
     [Load Balancer]
            ↓
     [Chat Servers]
       ↓        ↓
   [Redis]   [Queue]
       ↓        ↓
 [Presence]   [DB]
            ↓
     [Object Storage]
```

---

# ⚖️ 15. Trade-offs

| Choice      | Pros      | Cons                   |
| ----------- | --------- | ---------------------- |
| WebSockets  | Real-time | Persistent connections |
| Polling     | Simple    | High latency           |
| Queue-based | Reliable  | Slight delay           |

---

# 🔥 Interview Power Lines

Drop these and you’ll stand out:

* “I’ll use WebSockets for persistent real-time communication”
* “Decouple message processing using Kafka”
* “Partition messages by conversation for scalability”
* “Store media separately and send URLs”
# 🧠 1. Requirements

### ✅ Functional

* Users can **post content** (text, image, video)
* Users can **follow/friend others**
* Show a **personalized news feed**
* Feed ordered by **recency + relevance**
* Infinite scroll (pagination)

---

### ⚡ Non-Functional

* Low latency (feed loads fast ⚡)
* Highly scalable (millions of posts/sec)
* High availability
* Eventually consistent (OK if slight delay)

---

# 🏗️ 2. Core Challenge

👉 The hardest part:

> “How do we efficiently deliver a personalized feed to millions of users?”

---

# 🧩 3. Two Main Approaches

## 🔹 1. Pull Model (Fan-out on Read)

* Fetch posts when user opens app

```id="1u3h1p"
User → Fetch posts from all friends → Merge → Sort → Show
```

✔ Simple
❌ Slow for users with many friends

---

## 🔹 2. Push Model (Fan-out on Write 🚀)

* When user posts → push to followers’ feeds

```id="5t6i3j"
User A posts → Push to followers’ feed cache
```

✔ Fast reads
❌ Expensive writes

---

## 🏆 Best Approach: Hybrid

* **Push for normal users**
* **Pull for celebrities (millions of followers)**

---

# 🏗️ 4. High-Level Architecture

```id="y6lgul"
Client → API Gateway → Feed Service → Cache → DB
                         ↓
                    Post Service
                         ↓
                     Fan-out Service
```

---

# 📦 5. Core Components

### 1. Post Service

* Stores posts in DB

### 2. Feed Service

* Builds user feed

### 3. Fan-out Service

* Distributes posts to followers

### 4. Cache (Redis)

* Stores precomputed feeds

---

# 🗄️ 6. Database Design

### Table: `posts`

| Field      | Type      |
| ---------- | --------- |
| post_id    | UUID      |
| user_id    | UUID      |
| content    | TEXT      |
| created_at | TIMESTAMP |

---

### Table: `followers`

| Field       | Type |
| ----------- | ---- |
| user_id     | UUID |
| follower_id | UUID |

---

### Table: `feeds`

| Field   | Type |
| ------- | ---- |
| user_id | UUID |
| post_id | UUID |

---

# ⚡ 7. Feed Generation Flow

### 🔹 When user posts

```id="5ec8ak"
1. Save post
2. Get followers list
3. Push post_id to followers’ feed cache
```

---

### 🔹 When user opens feed

```id="h1h9zm"
1. Fetch from cache
2. If missing → DB fallback
3. Return top N posts
```

---

# 🚀 8. Scaling Strategy

### 🔹 Problem: Celebrity users

Example:

* Cristiano Ronaldo posts → millions of followers

❌ Push model fails

✔ Solution:

* Don’t push
* Fetch dynamically (pull model)

---

### 🔹 Problem: Huge feed size

Solution:

* Store only recent posts (last 500)
* Use pagination

---

# ⚡ 9. Ranking Algorithm

👉 Not just time-based

Use:

* Recency
* Likes/comments
* User interaction

Basic idea:

```id="6v2kbg"
score = w1*time + w2*engagement + w3*relevance
```

---

# 📦 10. Media Handling

* Store images/videos in object storage
* Serve via CDN

---

# 🔄 11. Caching Strategy

* Cache feeds in Redis
* Cache hot posts
* TTL-based invalidation

---

# 🔒 12. Reliability & Consistency

* Eventual consistency (OK)
* Use queues for fan-out
* Retry failed deliveries

---

# 🧩 13. Final Architecture

```id="8xt6ha"
        [Client]
            ↓
     [API Gateway]
            ↓
     [Feed Service]
        ↓       ↓
   [Cache]    [DB]
        ↓
   [Fan-out Service]
        ↓
     [Queue]
        ↓
   [Workers]
```

---

# ⚖️ 14. Trade-offs

| Approach   | Pros          | Cons         |
| ---------- | ------------- | ------------ |
| Push model | Fast reads    | Heavy writes |
| Pull model | Simple writes | Slow reads   |
| Hybrid     | Balanced      | Complex      |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out instantly:

* “I’ll use hybrid fan-out (push + pull)”
* “Cache feeds for fast reads”
* “Handle celebrities using pull model”
* “Use ranking instead of chronological ordering”
# 🧠 1. Requirements

### ✅ Functional

* Upload videos
* Stream videos (adaptive quality)
* Search videos
* Like, comment, subscribe
* Recommendations (optional)

### ⚡ Non-Functional

* Very high availability
* Low startup latency (video starts fast ⚡)
* Scalable (millions of uploads, billions of views)
* Durable storage (no data loss)

---

# 🏗️ 2. High-Level Architecture

```id="h4o6g0"
Client → API Gateway → Video Service → Processing Pipeline → Storage → CDN
                          ↓
                       Metadata DB
```

### Core Components:

* Upload service
* Video processing service
* Metadata service
* Storage (videos)
* CDN (delivery)
* Search & recommendation systems

---

# 📂 3. Key Idea (VERY IMPORTANT)

👉 Separate:

* **Metadata (DB)** → title, views, etc.
* **Video files (Object Storage)** → actual content

---

# ⬆️ 4. Video Upload Flow

```id="qv3h8m"
1. User uploads video
2. Generate pre-signed URL
3. Upload directly to storage
4. Save metadata in DB
5. Trigger processing pipeline
```

---

# 🎬 5. Video Processing Pipeline

After upload:

```id="3q5a1g"
Raw Video → Transcoding → Multiple Formats → Thumbnail Generation
```

### Why?

* Different devices need different formats
* Adaptive streaming (360p, 720p, 1080p)

---

# ⚙️ 6. Transcoding

Convert video into multiple resolutions:

```id="t7xk7q"
1080p → 720p → 480p → 360p
```

👉 Use async workers (queue-based system)

---

# 📦 7. Storage Design

### Object Storage (like S3)

```id="7qekxq"
video_id → stored in chunks
```

### Why chunking?

* Faster uploads/downloads
* Resume support

---

# 🌍 8. Content Delivery (CDN)

👉 Use CDN for global delivery

Flow:

```id="fnp8ry"
User → CDN → nearest edge server → video stream
```

✔ Reduces latency
✔ Handles huge traffic

---

# ▶️ 9. Video Streaming

Use protocols:

* HLS (HTTP Live Streaming)
* DASH

### Adaptive Bitrate Streaming:

```id="2v7s1h"
Auto switch quality based on network speed
```

---

# 🔍 10. Search System

* Index metadata (title, tags)
* Use search engine (like Elasticsearch)

---

# 👍 11. Engagement System

* Likes, comments, subscriptions
* Stored in separate scalable services

---

# ⚡ 12. Scaling Strategy

### 🔹 Problem: Huge traffic

Solution:

* CDN caching
* Load balancing

---

### 🔹 Problem: Large storage

Solution:

* Distributed object storage
* Data replication

---

### 🔹 Problem: Processing load

Solution:

* Async queues (Kafka)
* Worker scaling

---

# 🔄 13. Caching

* Cache popular videos
* Cache metadata
* CDN edge caching

---

# 🔒 14. Security

* Authentication (JWT)
* Private/unlisted videos
* DRM (optional for premium content)

---

# 📊 15. Analytics

Track:

* Views
* Watch time
* Engagement

👉 Use event streaming pipeline

---

# 🧩 16. Final Architecture

```id="s5a0k5"
        [Client]
            ↓
     [API Gateway]
            ↓
     [Video Service]
        ↓       ↓
   [Metadata DB] [Queue]
                      ↓
              [Processing Workers]
                      ↓
              [Object Storage]
                      ↓
                   [CDN]
```

---

# ⚖️ 17. Trade-offs

| Decision       | Pros            | Cons         |
| -------------- | --------------- | ------------ |
| Pre-processing | Smooth playback | Storage cost |
| CDN            | Fast delivery   | Expensive    |
| Chunked upload | Reliable        | Complex      |

---

# 🔥 Interview Power Moves

Say these lines and you’ll sound solid:

* “I’ll use pre-signed URLs to offload uploads”
* “Transcoding pipeline for adaptive streaming”
* “CDN for low-latency delivery”
* “Separate metadata and video storage”

# 🧠 1. Requirements

### ✅ Functional

* Show suggestions as user types
* Suggestions based on:

  * Prefix match
  * Popular queries
  * User history (optional)

---

### ⚡ Non-Functional

* Ultra low latency (**<100 ms ⚡**)
* Highly scalable (millions of queries/sec)
* High availability
* Real-time updates (trending queries)

---

# 🏗️ 2. High-Level Architecture

```id="y7p9bo"
Client → API Gateway → Autocomplete Service → Cache → Data Store
```

---

# 🔑 3. Core Problem

👉 “Given a prefix, return top K matching queries fast”

---

# 🧩 4. Data Structure Choice (VERY IMPORTANT)

## 🏆 Trie (Prefix Tree)

This is the backbone.

### Example:

![Image](https://images.openai.com/static-rsc-4/kHPxnu9sucdql8TcMy6-WicCxIg6uL14W3EpeBS0qmzdt8U0ys4N3GYm_m8yb4d1QbpwooM3d2OCk7ZzIX9KLAR8EwL9J2s-V_oQZpkayYiAM9gySReCRknGGAte2k0sKNWo6iLHXZg8raQiguzUynsxva0nnjWunGOZzt1yptU_sFtRRRjhXGTjvDUO9Ajc?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/iLPb4iH6YrXVTNhjw6ZUhhLi0bfZlv5NABDqwtf3Cw_JEB3LzMN-uJxBM8XBbtqnRb6TfdjBdQMeRpzmjP2WKINgknsRiCY0yhNnmXFZmium8hu7fFz5lQ52en2sh6cDn-48NuW5weE7arJmLYW4Z7a2sLk25IoMQe7d1r23RbitNAOOgph7aOyeQQAL6EYQ?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/cdiAs9DAAF4t3Iju_q38x_6iwR5PdvKYFxC-Brhzm2muJo1Bpn_hgVY8wiel7k1CbiE1EQFhmF8L5rYnr0AZg0fzLUjJI0Bsnd2kTZEq74oQ1BO3Le6_Qw1OnCDrmsSBqrDxnH6EpFdvJpGsTjn9ygVYDCMPttF2i-bW_d7KcKVoPjnaUsWAbgZvQ-JP_swk?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/DdsBq0bdhi3DolAlbB5xvowMU6g7vcxPPs_C_gjPlftc5VkZXNAfxCDeHWgj1tqcg34PyzmE-rvEmu3Vxqe7qPmE7wa4D8q0yGF8N0nveYCYlyRvLqPa1u99nec5gUgDEuoREB4zdNH2lkaolshAmaZsCpkd2oGJIyulGRCp-mRA6KPcLVudgbAYb3ILK7BG?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/pvENOjj42ZPwxlFeuNwUXlYMowUQEYQV2zz4XkIuYE0GP9ZLeuNHzrxTFpEcLl1outVamwjWgKQnjN4weWp5DvtvktZn45oyvnQ9XvEu3kmtVUQlCzrLdOXxlFMeVjUhOqTuWwrv_35G7KXm0ZGE4OQVFYkwPlT_YICNYMdTn_owkvhI0fbCL-g-0n99uP96?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/sHtXwJeCRAjMNXmSnnvnqC9WZc3C-knZMCeGMgxYc-BYl5en0NU-n5i-WFKwpVrJen5BVFtEeq3Lhv4CDquseGiIn611UrJKURxltRRhzvx7IwzTOHyjc3LXkkICP44oYI_eMaQbo1okM_BS6ncshWrEMRLaoKIMWgB3j-6fLq-DslYq1SOKYySSaaHQ-IJY?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/XvbZDw5Kgd1VhaCIXeidWipCg6nxd1TMNPRLXcl3Ir8LNIhNhIhplQgXbdMYqGIWYuBt8Ny3CPtiesfyKFNt9D6KPksU3mPkSbOABS3D8FbD7xb8KvYVVIdaADwsmZLPGjyf7o2RVzkvafn_vDdLRE2LonEYdBvmt9wtLg4WN3JObF_5j35L7MZb6cLXfeDF?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/1A9GTvbdgLzQ3u3f_iZN5UJup98Do_mj6sFYSAn9tzy6PISHeiYRNiB4ijMDpOuCjtYbPHK7LW_nPh1A-l22ISAxxHl9ER7XpQe00dzQTEwsPH9bPP_d1CdDnbZZf-FXTlsU9QaiKUvbUxno2NzLP1tkNF_eaE1Dh9zj8HMN8kgXh05Yn5eSHu-sTQcc6T5v?purpose=fullsize)

* Each node = character
* Path = word/query

---

### Why Trie?

✔ Fast prefix search → **O(prefix length)**
✔ Efficient for autocomplete

---

# ⚙️ 5. Storing Top Suggestions

Each node stores:

```id="6o2f3m"
top_k = ["apple", "amazon", "airpods"]
```

👉 Precompute top suggestions for fast lookup

---

# ⚡ 6. Query Flow

```id="qg6f91"
User types "app"
→ Traverse Trie
→ Fetch top_k suggestions
→ Return results
```

---

# 📊 7. Ranking Logic

Not just alphabetical ❌

Use:

* Frequency (how often searched)
* Recency (trending queries)
* Personalization (user history)

Example:

```id="1rf2c9"
score = w1*frequency + w2*recency + w3*user_preference
```

---

# 🚀 8. Scaling the System

### 🔹 Problem: Huge dataset (billions of queries)

Solution:

* Shard Trie by prefix (a–z)
* Use distributed storage

---

### 🔹 Problem: Low latency

Solution:

* Cache hot prefixes in **Redis**
* Keep Trie in memory

---

# 🔄 9. Updating Suggestions

### Batch Updates

* Periodically rebuild Trie (daily/hourly)

### Real-Time Updates

* Stream trending queries
* Update top_k dynamically

---

# ⚡ 10. Caching Strategy

```id="z3d4lm"
Key: "app" → ["apple", "app store", "application"]
```

✔ Avoid recomputation
✔ Serve frequent queries instantly

---

# 🔒 11. Edge Cases

* Typo tolerance (fuzzy matching)
* Multi-language support
* Offensive content filtering

---

# 🧩 12. Final Architecture

```id="xkz1n6"
        [Client]
            ↓
     [API Gateway]
            ↓
 [Autocomplete Service]
      ↓          ↓
   [Cache]    [Trie Store]
```

---

# ⚖️ 13. Trade-offs

| Approach      | Pros               | Cons         |
| ------------- | ------------------ | ------------ |
| Trie          | Fast prefix search | Memory heavy |
| DB LIKE query | Simple             | Slow         |
| Cache-heavy   | Ultra fast         | Memory cost  |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use Trie with precomputed top-K results”
* “Cache hot prefixes for sub-100ms latency”
* “Use ranking based on frequency + recency”
* “Shard Trie for scalability”

# 🧠 1. Requirements

### ✅ Functional

* `GET(key)` → value
* `SET(key, value)`
* `DELETE(key)`
* TTL (expiry support)

---

### ⚡ Non-Functional

* Extremely low latency (**<1 ms ⚡**)
* High throughput (millions of ops/sec)
* Scalable (horizontal scaling)
* Fault-tolerant

---

# 🏗️ 2. High-Level Architecture

```id="jzdl3z"
Client → Load Balancer → Cache Nodes (Cluster)
```

Each node = in-memory store

---

# 🔑 3. Core Idea

👉 Store everything **in RAM**

Why?

* RAM is much faster than disk
* Enables ultra-low latency

---

# ⚙️ 4. Data Storage Design

Each node maintains:

```id="7cix28"
HashMap<Key, Value>
```

Optional:

* TTL metadata
* LRU tracking

---

# ⚡ 5. Data Partitioning (Very Important)

We need to distribute data across nodes.

## 🏆 Consistent Hashing

![Image](https://images.openai.com/static-rsc-4/I8GY-ewhxAWuq7wqme8BVsVNY2bPseXbBB9kpxKgnRs3mdNYbh83U4aBrZGAC7sKGkytezKwwStwq5DYh1JIEuXY-Vpz2EcH5pEBFjGMYuhXRloy7y8B3pX6qGCfBVjhMKpOc1PwoBhBZsnui4ovAeSAs2caN1d9RXGIGVuSojbMBZoAOIdI98qkHpPNn-yf?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/9R5gGe4r0utEOPWggM4zC7-S7GJIBGkHUywqmiETpjLyQVk0Q-d-ZJd9hw7XW_pstLBt12VLtHIVK-eQSTzE4ZeFK1VHwAm8cQXF8I05dXeEhN3sY7GW2cTyEjmD7knbyX0wbZtDidsB2HFPJLqghVKHhqwAn1k5LVHaZF5TeDQWba4GyAK4tGFXEWkH-X6P?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/VFV3s410BAp9Qo-DCsKezo7tC5ebktNZ67KYew9hut5UHucsVuH_XlJqnd3HIqRRoUoFLC_WDarfCPgbL2KB1FNmPzsvsr_RdrG4OeaREdJLBlJ1J7uSn4FtESTbJXJPaYirM2UWRH9fNATZqcVaHKszs49p9UH0tkPNH7ViGZwST1wzOW645xr9N6a7SEsA?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/of173afivOo4U9ZUG1f2ox-m1TrsqScGW81XDJikcrKT9ozIB9a5TChWlwlBUOBsJA8-1u7CQSILb_VXISA09KhK-i4aPR7QodkQrbz2G_qaiNQN8o3IM-CsEH0i9o_AaFX18wF_vnLpLCmxL241wTQ6qsnh9JYZbNYNg-R5RJ1z3u6QVRBf72ouBpVWOcWi?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/nkPZdvNSRa-Tx3TSNPGodxxgsfEyeeAd0wMDif-HEpgteQp3AEIs4S3WeO1ManoFUxcI-aXdzuKJRHtRpk_8rlGcBdz0XM_Dll834VHRpK_liaQvxjfrkV0VBcZb85AV-yBhz87p5ISYhRDN-ez-SKBK_Qpk0bMrJxCP-9Ha_DMzNjbfjYjxn-1Iz_BDm_1-?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/YrmDABUeSA4-UVNOEgynuV4G_WGo-R6gOGKPzGz8QOmJ2N7Swci1yZ0MUJ18QXYQa3VoViKctKmHbs0SSLxcgC3dAQZtCFLW3UzavlUi-7PNpeMUJ1sPWrU6pOqxqvncph_uXfa4L092lrQVPhc00deVUPEQBizFgiFKBjF7DnxZ9ooIg5GJtQQly9srumb3?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Nn3y11w2x1PV0zKGJQrMDa3f0UTpjMqFv6LujV_GnoW72f9xSCeefsV6jrb9m18GntvrHcmXShDFLos0I1P9-YFvU8M3VLBLJK86tf9Ndu-E7MGlR0SjWA6ySIFtZsIwEnGnnocjIPi2uzgD4ApA34NaPQeGOJD1Nmd0TAS0fyvdZxR04IPLDRK9uyKWde7x?purpose=fullsize)

### How it works:

```id="z3shg6"
Hash(key) → map to node on ring
```

✔ Even distribution
✔ Minimal rebalancing when nodes added/removed

---

# 🔄 6. Replication (Fault Tolerance)

Each key stored in:

* Primary node
* Replica nodes

```id="z0gk6x"
Node A → primary  
Node B → replica  
```

✔ Prevent data loss
✔ Improve read performance

---

# ⚡ 7. Read & Write Flow

### Write:

```id="v5v9pq"
SET(key)
→ Hash → Node A
→ Store + replicate
```

### Read:

```id="5pq1t6"
GET(key)
→ Hash → Node A
→ Return value
```

---

# 🧮 8. Eviction Policies

When memory full:

* LRU (Least Recently Used) 🏆
* LFU (Least Frequently Used)
* TTL-based eviction

---

# ⏳ 9. Expiry Handling

Each key has TTL:

```id="az3g2z"
key → value + expiry_time
```

Strategies:

* Lazy deletion (on access)
* Background cleanup thread

---

# 🚀 10. Scaling the System

### 🔹 Add more nodes

* Consistent hashing minimizes data movement

### 🔹 Handle hot keys

* Replicate hot keys
* Use local caching

---

# ⚡ 11. Performance Optimizations

* Use **event-driven I/O (non-blocking)**
* Multi-threaded or async processing
* Pipeline requests (batching)

---

# 🔒 12. Consistency Model

Options:

* Strong consistency (slow)
* Eventual consistency (faster 🚀)

👉 Most caches prefer **eventual consistency**

---

# 🧩 13. Final Architecture

```id="d3n7dn"
        [Client]
            ↓
     [Load Balancer]
            ↓
   ┌─────────────────┐
   │ Cache Cluster   │
   │  Node A Node B  │
   │  Node C Node D  │
   └─────────────────┘
```

---

# ⚖️ 14. Trade-offs

| Design Choice      | Pros       | Cons              |
| ------------------ | ---------- | ----------------- |
| In-memory          | Ultra fast | Limited size      |
| Replication        | Reliable   | Extra memory      |
| Consistent hashing | Scalable   | Slight complexity |

---

# 🔥 Interview Power Moves

Say these and you’ll sound strong:

* “I’ll use consistent hashing for partitioning”
* “Replication for fault tolerance”
* “LRU eviction to manage memory”
* “Eventual consistency is acceptable for cache”
# 🧠 1. Requirements

### ✅ Functional

* User can **request a ride**
* Match with **nearby drivers**
* Real-time **location tracking**
* Trip lifecycle:

  * Requested → Accepted → In Progress → Completed
* Fare calculation & payments

---

### ⚡ Non-Functional

* Low latency (matching must be fast ⚡)
* High availability
* Scalable (millions of rides/day)
* Accurate location tracking

---

# 🏗️ 2. High-Level Architecture

```id="gtt9i1"
Client → API Gateway → Ride Service → Matching Service → Driver Service
                          ↓
                      Location Service
                          ↓
                        Database
```

---

# 🧩 3. Core Components

### 1. Ride Service

* Handles ride requests
* Maintains trip state

### 2. Driver Service

* Manages driver availability

### 3. Location Service 🚨 (VERY IMPORTANT)

* Tracks real-time GPS data

### 4. Matching Service

* Finds nearest driver

---

# 📍 4. Location Tracking

Drivers send location updates:

```id="nfh3ya"
Every 2–5 seconds → lat, long
```

Store in:

* In-memory DB (Redis / GeoHash)

---

# 🗺️ 5. Geo-Spatial Indexing

## 🏆 GeoHash

![Image](https://images.openai.com/static-rsc-4/hXxORExcxWcr2xLNEEeeiJUnji8LMOfx24Y2-ZlR6lZWXZLW9JiKu1mvZWZaEg84z2uAifXrbYZRT_sXTCGWnRa5kverKiVKFVGLWGbSOpQP6ApP6zkWDf6NP-nblK2jCGji4d5ePO9n0FF73BSLek4bC8LnGyo8eEZKyZL0aG7riXLY3-ukh89ZS3HxOFjT?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/L2sie55E07z59KEDguYy6EUlTPsrIoaxeelb_dOLy9rv7ZAt7L3EsNcScWrkMUQYHiYQnF2cLQ-kkMMKnBhH3Y-JxZWhB7dmwo80BaZAaYvY9fdjFzxhqFiwW2ZMtXM5nREarpoEKrQTGG7stc8Q6KSmB-xC4RsGAtamgzXuVR3tq60rMpiPQKfaLi8PKtpy?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/fGTvzDmk1e0_Rsyh6_i0k2saDAbZG_4_dK0xeC08uR-PIRX47Y1zGdcRr6Ar7RoHwT_LaIEvGD39L9Iuxf0akoEtDQF-ebx6SSQG9aWXx4us-6OaOYFn10c4doCdYuCnzS2EvDrPvr14oBisoTnPC2o1ig3c4KxkaUhJFpR_nmU1BIitZbUidVa6UKdo5Fhs?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/7cx-j_4hW1L0XKcK0LgwCHb9L8dv-TD_XaLrSm409seVlLlqkeTa9YM89tufFGZwUAo_m4nab1ior2sOA2L9k_exVbdxbLq0xW-zLp61z8oDcgVeRemqSKePwTkUfSAkKGZI3iHdcrwhnJ0ThE-P9nmpexTkQeRVeOY0KA8csQ8pFCdvnBU3hJ3992eQ3FSe?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hVbyVUB-oQOXYwvQazrup1KzJ8C_fX03TRq06PFuwQcAAvQeoEkZqeWpKAUov_whQ8Z12eF__xZF-MtbWgf80TYOQg8qFqUueEzSpg-AQmE_bRi1HagifMrBHbRtESUrmr4Q4h_aZImvKcmoadKLpmAk2cMU-PQfOeACG4wrORImA4rNe5dccxzObnlXOYxX?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/ifu5PIAshJ3QOylsSI_194tlpSZxBkAyedUolkg4mPzx4QZ85_1R6EMGFbkU-xage23LI-eRZh7gDX-eXUsbfRt4RQPu0ywlnLHdlZ6G3KlQI97kXE4fChCDfrLAOXkmLuBnV9JqDHsw_Hr9ckzD6ljVxiYhqXvupYZQks3apX7IBsqEX-ILDvY3_ZYqe-9G?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/18OrJ_5jHP65jYMnzA2H9XzdGVE0P982kbeHSeBK4mU1k2PxVIZFrqgv-b5WbpaBUSaxfkKOdqyMmbPwhIJQC6pRTmrdHEcFMVJlq-pss-cSPCwOmYn2Pwg5g3c97q9Z_HZozgKUfGbMcGrmWUzZK6zm9DqQR_qo4KAC0Q4BoXGJtiRyKliHWsviN0zZFCMV?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/cOCTyw_y8WEnI8CobbOJLexjQDjNfoXwMI-jWuBua9qjbvhSWvIvh8Fl5chVbPpLhVADQshq1kzwVVJ6Ykpt0bclozyJ58v7FqP9yBUe6QzAKj2WwPhMkyKsEnsoq3RdH_1QvXGlBTh8UkeThmhhYk5anFtapmRIQQUxibqD9Z2yZI1R7R_lJADhyV_R4Hkm?purpose=fullsize)

👉 Convert lat/long → string
👉 Nearby drivers = similar prefix

---

# 🚖 6. Matching Flow

```id="zn4gnx"
1. User requests ride
2. Get user location
3. Find nearby drivers (GeoHash)
4. Rank drivers (distance, rating)
5. Send request to top drivers
6. First accept → assign ride
```

---

# ⚡ 7. Real-Time Communication

Use:

* WebSockets / MQTT

For:

* Driver updates
* Ride status updates

---

# 💰 8. Pricing System

Factors:

* Distance
* Time
* Surge pricing (demand > supply)

---

# 🚀 9. Scaling Challenges

### 🔹 Problem: High location updates

Solution:

* Use in-memory store (Redis)
* Batch updates

---

### 🔹 Problem: Matching latency

Solution:

* GeoHash indexing
* Precompute nearby drivers

---

### 🔹 Problem: Surge demand

Solution:

* Dynamic pricing
* Queue ride requests

---

# 🔄 10. Trip Lifecycle

```id="5gwd9j"
Requested → Matched → Driver Arriving → Trip Started → Completed
```

---

# 🔒 11. Reliability

* Retry matching if driver cancels
* Timeout for driver response
* Store trip state in DB

---

# 🧩 12. Final Architecture

```id="vdy9r7"
        [Client]
            ↓
     [API Gateway]
            ↓
     [Ride Service]
       ↓       ↓
 [Matching] [Location Service]
       ↓         ↓
   [Queue]    [Redis]
       ↓
   [Driver Service]
```

---

# ⚖️ 13. Trade-offs

| Design Choice     | Pros              | Cons                |
| ----------------- | ----------------- | ------------------- |
| GeoHash           | Fast proximity    | Approximate results |
| Real-time updates | Accurate tracking | High load           |
| Push matching     | Fast assignment   | Complex logic       |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use GeoHash for efficient nearby driver lookup”
* “Use Redis for real-time location tracking”
* “WebSockets for live updates”
* “Matching based on proximity + driver rating”

# 🧠 1. Requirements

### ✅ Functional

* Browse/search products
* View product details
* Add to cart
* Checkout & payment
* Order tracking
* Seller/product management

---

### ⚡ Non-Functional

* High availability (always up)
* Low latency (fast browsing ⚡)
* Scalable (millions of users/products)
* Consistent payments (no double charge 🚨)

---

# 🏗️ 2. High-Level Architecture

```id="c1u3qt"
Client → API Gateway → Microservices → DB / Cache / Queue
```

### Core Services:

* User Service
* Product Catalog
* Search Service
* Cart Service
* Order Service
* Payment Service
* Inventory Service
* Notification Service

---

# 🧩 3. Core Components

## 🛍️ Product Catalog Service

* Stores product info

### Table: `products`

* product_id, name, price, description, category

---

## 🔍 Search Service

* Full-text search (Elasticsearch)

```id="1b9pxf"
Search "laptop" → ranked results
```

---

## 🛒 Cart Service

* User’s temporary items

Stored in:

* Redis (fast access)

---

## 📦 Order Service

Handles:

* Order creation
* Order state

```id="7m4kgh"
Created → Paid → Shipped → Delivered
```

---

## 💳 Payment Service

* Integrates with gateways (like Stripe)

👉 Must ensure:

* Idempotency (no duplicate payments)
* Transaction safety

---

## 📊 Inventory Service

* Tracks stock

```id="8hm31z"
product_id → available_quantity
```

---

# 🔁 4. Checkout Flow (VERY IMPORTANT)

```id="3l2g3f"
1. User clicks checkout
2. Validate cart
3. Check inventory
4. Process payment
5. Create order
6. Reduce inventory
7. Send confirmation
```

---

# ⚡ 5. Key Design Challenges

## 🔹 1. Inventory Consistency 🚨

Problem:

* Multiple users buying same item

Solution:

* Use **locking / atomic DB operations**
* Or **eventual consistency + compensation**

---

## 🔹 2. Payment Reliability

* Use idempotency keys
* Retry safely

---

## 🔹 3. High Traffic

Solution:

* Cache product data (Redis)
* CDN for images

---

# 🚀 6. Scaling Strategy

### 🔹 Read-heavy system

* Cache product pages
* Use CDN

---

### 🔹 Write-heavy parts (orders)

* Use queues (Kafka)
* Async processing

---

# 🔄 7. Microservices Communication

* Sync (REST APIs)
* Async (message queues)

---

# 📦 8. Caching Strategy

* Product cache (Redis)
* Cart cache
* Search results cache

---

# 🔒 9. Security

* HTTPS everywhere
* Secure payments
* Authentication (JWT/OAuth)

---

# 📊 10. Analytics

Track:

* User behavior
* Sales trends
* Recommendations

---

# 🧩 11. Final Architecture

```id="3sjz8q"
        [Client]
            ↓
     [API Gateway]
            ↓
 ┌─────────────────────┐
 │ Microservices Layer │
 └─────────────────────┘
   ↓    ↓    ↓    ↓
[DB] [Cache] [Queue] [Search]
```

---

# ⚖️ 12. Trade-offs

| Decision         | Pros       | Cons                 |
| ---------------- | ---------- | -------------------- |
| Microservices    | Scalable   | Complex              |
| Caching          | Fast reads | Stale data risk      |
| Async processing | Reliable   | Eventual consistency |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use microservices for independent scaling”
* “Cart stored in Redis for fast access”
* “Ensure idempotent payments”
* “Use event-driven architecture for order processing”
# 🧠 1. Requirements

### ✅ Functional

* Browse restaurants & menus
* Search/filter (cuisine, rating, distance)
* Place orders
* Real-time order tracking
* Assign delivery partner
* Ratings & reviews

---

### ⚡ Non-Functional

* Low latency (fast browsing & checkout ⚡)
* High availability
* Scalable (peak meal-time spikes)
* Real-time tracking accuracy

---

# 🏗️ 2. High-Level Architecture

```id="2m2p6u"
Client → API Gateway → Services → DB / Cache / Queue
```

### Core Services:

* User Service
* Restaurant Service
* Search Service
* Order Service
* Delivery/Driver Service
* Location Service
* Notification Service

---

# 🧩 3. Core Components

## 🍽️ Restaurant Service

* Stores restaurant & menu data

### Table: `restaurants`

* id, name, location, rating

---

## 🔍 Search Service

* Full-text + geo-based search

```id="0q9b4y"
Search "pizza near me" → ranked results
```

---

## 🛒 Order Service

Handles order lifecycle:

```id="t3w1vt"
Created → Accepted → Preparing → Out for delivery → Delivered
```

---

## 🚴 Delivery Service

* Manages delivery partners
* Tracks availability

---

## 📍 Location Service 🚨

* Tracks:

  * Delivery partner location
  * User location

Stored in:

* In-memory DB (Redis with GeoHash)

---

# 🗺️ 4. Nearby Search (Geo Queries)

## GeoHash / Geo-indexing

![Image](https://images.openai.com/static-rsc-4/7cx-j_4hW1L0XKcK0LgwCHb9L8dv-TD_XaLrSm409seVlLlqkeTa9YM89tufFGZwUAo_m4nab1ior2sOA2L9k_exVbdxbLq0xW-zLp61z8oDcgVeRemqSKePwTkUfSAkKGZI3iHdcrwhnJ0ThE-P9nmpexTkQeRVeOY0KA8csQ8pFCdvnBU3hJ3992eQ3FSe?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/9Jw_OjhZvKnplydyxHy_ywfeKfYentLayE9fpGKUboh3Hm4xsqDF3zWsmHYEoxCQdN8MEwtQeZmQst52pm68Z8lyridjFLlVU7aXojLUueA-V7nb38x2aDLbxiR3FJGGtw9Qjbh2VltI75MTYVn3xSlE2joqEVGg6ogOV2a2lfBb3-772FZsmjNsqHJ_hOWT?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/2YzeUkQxz33IGzhR_8PG2lOlHPFTGKlFdPjhXzqZNiJWRudxSMafMcc8unG0z7-Wl_ZE_dfvZz3UPaTqU3USTLerh1GWpzxcMd5B5-LEcVWbNe_Q0ovAS0Wl7qMmDqwWFN7Sw0w716naDz4gaC_v-fyuUP-LFK2VaG6Bl9Lrw8SMNnrZ6WUP2lkqRplaUJnO?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/ifu5PIAshJ3QOylsSI_194tlpSZxBkAyedUolkg4mPzx4QZ85_1R6EMGFbkU-xage23LI-eRZh7gDX-eXUsbfRt4RQPu0ywlnLHdlZ6G3KlQI97kXE4fChCDfrLAOXkmLuBnV9JqDHsw_Hr9ckzD6ljVxiYhqXvupYZQks3apX7IBsqEX-ILDvY3_ZYqe-9G?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/6meUuiCO9Wdd37KyxPGxzVXIAe6nJuOyuqLFFXsewoeVES0e9waNM7KeQVKZuM8ydpqNUlnbzCBrmTlESmS_036JC3xT2Tyc5FZgDTweGaYxQ26HACCiEjV1YmpVcy97VoR9A2NjclqCIiRBVWcpl2700ltHvi8XNmwNnTmOHrweTiZ4Oaa1QzJQKRcIxyeg?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4giPagenqTK_UfeOGFK3XvtM44iyFvO0GQjX8I4iYNTKu-JhG7zgmBrQl_0qrlratNAicUjx_QcrMYrRZPOqC6TC-YzepcEO4W3umjTDWs9VZKmsxq4oVNMd6lfdqpWSTV7vCbqw_10I6ig6O7qzFdQtJ27VkjiTol01fk8wT-H1izxwohtbid6yLuKBaRAO?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/gHNWiX7bEuOK7EzasLPK30r0prMedi5CuYU-CReI5mgDgNKUGJLYQoJBtaoUKKFC7QhpizU_LxVxwzEHrnXcpF4d5bUInBB6SWIcWEjzv5alrpqnCs1ZAmDh-Mgg4m5F8eQIuFyo7u4dvtNSw6SnzmG3gxPcLnAUjblvKdMC62YudFTM-CGALO8Bg3FbiFTF?purpose=fullsize)

👉 Convert lat/long → grid
👉 Quickly find nearby restaurants/drivers

---

# 🔁 5. Order Flow (VERY IMPORTANT)

```id="4z4qhv"
1. User places order
2. Validate restaurant & items
3. Create order
4. Notify restaurant
5. Assign delivery partner
6. Track order status
7. Deliver to user
```

---

# 🚀 6. Delivery Assignment (Core Logic)

```id="n9s7fs"
1. Find nearby drivers
2. Rank by distance + availability
3. Send request
4. First accept → assign
```

---

# ⚡ 7. Real-Time Tracking

* Use WebSockets / MQTT
* Driver sends location every few seconds

```id="3s3ptq"
Driver → lat/long updates → user sees live map
```

---

# 💰 8. Pricing System

* Base delivery fee
* Distance-based pricing
* Surge pricing (peak hours)

---

# ⚡ 9. Scaling Challenges

### 🔹 Peak traffic (lunch/dinner)

Solution:

* Auto-scale services
* Queue-based order processing

---

### 🔹 High read traffic (menu/search)

Solution:

* Cache menus (Redis)
* CDN for images

---

### 🔹 Real-time updates

Solution:

* Use in-memory store
* Event-driven system

---

# 🔄 10. Caching Strategy

* Cache restaurant lists
* Cache menus
* Cache search results

---

# 🔒 11. Reliability

* Retry order assignment if driver cancels
* Timeout handling
* Store order state in DB

---

# 🧩 12. Final Architecture

```id="d6z3cu"
        [Client]
            ↓
     [API Gateway]
            ↓
 ┌─────────────────────┐
 │ Microservices Layer │
 └─────────────────────┘
   ↓    ↓    ↓    ↓
[DB] [Cache] [Queue] [Search]
```

---

# ⚖️ 13. Trade-offs

| Decision           | Pros        | Cons            |
| ------------------ | ----------- | --------------- |
| Real-time tracking | Better UX   | High load       |
| GeoHash            | Fast search | Approximate     |
| Caching            | Fast reads  | Stale data risk |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use GeoHash for nearby search”
* “Real-time tracking via WebSockets”
* “Queue-based order processing for scalability”
* “Cache menus and restaurant data”
# 🧠 1. Requirements

### ✅ Functional

* Live video streaming
* Real-time chat alongside stream
* Follow/subscribe to streamers
* Reactions (likes, emotes)

---

### ⚡ Non-Functional

* Ultra low latency (live stream delay minimal ⚡)
* High scalability (millions of viewers)
* Reliable (no stream drop 🚨)
* High throughput (chat can be crazy fast)

---

# 🏗️ 2. High-Level Architecture

```id="3n6d6y"
Streamer → Ingest Servers → Processing → CDN → Viewers
                                 ↓
                             Chat Service
```

---

# 🎥 3. Video Streaming Pipeline

## Step 1: Ingestion

Streamer sends video using:

* RTMP protocol

```id="9g4d7w"
Streamer → Ingest Server
```

---

## Step 2: Transcoding

Convert stream into multiple qualities:

```id="6v2n2o"
1080p → 720p → 480p → 360p
```

---

## Step 3: Distribution

![Image](https://images.openai.com/static-rsc-4/0PnthXLMAHb1YwvVXlcHPu9of05hqF-0b_qAEFB7lj93Avmy8eWuIuIPTpCRU2gGPolFbays4pYtUP9eRmnRqxY4myKz2Gj9f0-FE87JoYquJozbNRxvX-DnWLqEb6zRIvdMffya8xzDwcpA9ZCcjZA4nQGtVvhUrCQHE4-KzZlOmPM9BjMJU1gCMBF7rpIv?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/qXGZGtQ44r73L-ZpUVTMq-OJMLG0Y2Lrj4D8bKDQUIkZemxWz5WJpTIeDIWva51Yh1Fc3RA-onf0pGWl8FeI5m-6rNN3lorw9LT8_bebThA0ezw4_cu6YwWPQYs4LJZPmXec0tza-5zx1Wx1U2uB5XWrwyqtQEOdyOQlBzfBgY-YtR_ETX2ayIT3ALpII3or?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/SG5iTfXc2ziefixQphKE62RlUE7BM1jQyFT2zxHRgGQc2PVryeJytRoiI4AIxRhpLUrBKAb50n_RDh6X3H40tKXmcZrJZS621BdXoK8Iili9PbjoGDuLUCWQ7Wt6aCtg4W31y-X0Zc1KyBWjhHgc9FqfoYunJ_heAz4BXstRPz4OH8--NqY5nOUve3eEnOOk?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/JwJNeZx8Xtwj3A4Fn9KkW8lfa5lR8Bt7exB3j-6t3VzTmchTt4ltCC3YBzI9BIl8jkOMYXnqhBMoyZ0wYGPTYhCYgVO_jmxZ6HOO624N0U83tV5TDJYD_BXhLq918yT9PVJCCV9TzMCIGu8mKHjk4ycjYECSXZRlMgtGapGIwpxx3CIgHAV9ZkNsdMEefQdn?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/tNikm5MwmaAvd8YIQ-Ae79OQnbDLQtdFUHvdAD1hYe7InntCdzaV1weHzEWcYCI9ixktVt2Ml_tDMcH9ZsRTOAZWudhUYdutDShzt3qOWnWKLcar83HMaVXPuB8mUDBZviM-kFLUBnAe_JFdxTUt_6BO21sTvca9uNyA5FQ4VnSd1P0Ug1K3c_a7dfkpEU5r?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/AF23msPlE0P4F7glc-KxWl6pN3MvWuO564ukwGnhK8I5Wyli8uO0qgtH-dOwMjkHYSoYklj5Y1IefyNgZvb2AiLxvtLE-ZeARbUwQHbNciqcY4XQjqpTBm77OIQRPL_UA5ADYhfML4jxrH7ATfsm86_uJgBoxnyhGYpPClqb-BnK5yvQqYi-cTfhTZ_OJ0Qq?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/71uIP3YJfPouq2ZInhGMQDO7n_krqbbV8gC2akxbWStPEn6YJdHUWT1juqQNFIY-7VdVpOMc946iYYsl8AbQUdeaPdxkxCrFVBkKqBKEhdOJPI7jPELhjz6wwkxB0IgtroDfocWYT1BnJ52D-MaPjCnCmg0W0XmDTr1r-DA_heDXRX9Pxo33RHd1OD0clXt0?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/GMzpnWahfCu0KkuD14YR0R0mBzkketlVKyLes9sCz3QTBw2cixtxKAGAL1SgOIHGNQEj2N57oNbn0KKmS-3hBo_TzWAIsbQcvVNd_RGKpTGnjig2OmcGPZCoMH0bKw-hPUxto62vLQPNkgqUllXcydMyuNpX9466Di8Ovs5PqcEZxQ_brznb9-Q5k5YyaSMR?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/9RW6zd8ECjaWMDMUMuVv3PUGgUItgdmoAYh9hKgTFWOaGgQdEq7CTrS6_OvKguvO4Gb2g5quZBlxdjwhOY3-wnfJVd19jYCNdc3qS_HlH3v_oHd4FkgPpAF8PwI46b7KiTpG9J9_dLeeTAjlDBuKxtTE7fgYzebM9AIutEFLB-uNvtZFdoz20gC6RAL_hTCX?purpose=fullsize)

* Use CDN for global delivery
* Use HLS/DASH protocols

---

# 💬 4. Chat System (VERY IMPORTANT)

Chat is a separate real-time system.

### Protocol:

* WebSockets (persistent connection)

---

# ⚙️ 5. Chat Flow

```id="z7y3bc"
User sends message
→ Chat server receives
→ Broadcast to all viewers
```

---

# 🚀 6. Scaling Chat System

### 🔹 Problem: Millions of messages

Solution:

* Partition chat by stream ID
* Use distributed chat servers

---

### 🔹 Broadcast Strategy

```id="5pt7x8"
Message → Chat server → Pub/Sub → All subscribers
```

Use:

* Kafka / Redis Pub-Sub

---

# ⚡ 7. Handling High Throughput

* Batch messages
* Rate limit users
* Use message aggregation (combine bursts)

---

# 📊 8. Chat Moderation

* Spam filtering
* Banned words
* Moderators/admin controls

---

# 🔄 9. Sync Between Video & Chat

👉 Important UX detail

* Slight delay in chat to match video latency
* Or sync via timestamps

---

# 📦 10. Storage

### Video:

* Temporary storage (live)
* Archive to object storage

### Chat:

* Store messages (optional)
* Use DB for history

---

# ⚡ 11. Scaling Challenges

### 🔹 Viral streams (millions viewers)

Solution:

* CDN handles video
* Chat sharded across servers

---

### 🔹 Low latency streaming

Solution:

* Use low-latency HLS
* Edge servers close to users

---

# 🔒 12. Reliability

* Multiple ingest servers
* Failover CDN
* Retry chat messages

---

# 🧩 13. Final Architecture

```id="v7l5z2"
        [Streamer]
            ↓
     [Ingest Servers]
            ↓
     [Transcoding]
            ↓
          [CDN]
            ↓
        [Viewers]

        [Chat System]
      ↓            ↓
 [Chat Servers] [Pub/Sub]
```

---

# ⚖️ 14. Trade-offs

| Component  | Pros               | Cons                   |
| ---------- | ------------------ | ---------------------- |
| WebSockets | Real-time chat     | Persistent connections |
| CDN        | Scalable streaming | Cost                   |
| Pub/Sub    | Fast broadcast     | Complexity             |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “Separate video and chat systems for scalability”
* “Use CDN for global video delivery”
* “Partition chat by stream ID”
* “Use Pub/Sub for real-time broadcasting”

# 🧠 1. Requirements

### ✅ Functional

* Serve static & dynamic content (images, videos, HTML)
* Cache content closer to users
* Reduce latency

---

### ⚡ Non-Functional

* Ultra low latency (serve from nearest edge ⚡)
* High availability (global outages unacceptable 🚨)
* Massive scalability (billions of requests)
* Fault tolerance

---

# 🏗️ 2. High-Level Architecture

```id="2pfj7f"
User → DNS → Edge Server (CDN) → Origin Server
```

---

# 🔑 3. Core Idea

👉 Move content **closer to users**

Instead of:

```id="6y3m7p"
India → US server (slow ❌)
```

Use:

```id="xg6u5f"
India → Nearby CDN edge (fast ✅)
```

---

# 🌍 4. Edge Locations

![Image](https://images.openai.com/static-rsc-4/F0_Tz5d7JECFrKSUuajZ_XAEHBWXwng7z09c4VZHoZR5DAFtpaHoa38E4kBydq-niV-eWoQYjqcg-F3hmiGlUxdUvd9VMqNe8xwFoC7vaLcMkTc-rg9HTvLQrF-D5_y7e40bkw39PzekKBOgjEndRz8Qalr5jzXyNZjYUTEeYa7y1tQVpymVC_BZ306I7cYc?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/jE832hm6l6lhK6Av9eKETnLErdsbVD9QNE9g6IRKM0HL3ZRnvT3x_y06Uo6mI45yxlTidhnn5ZZjzbDOvNbp4FXtaeblwa4kVuvDVTwDWIWBs5sNdTfcHzsMOQwMQRzHd73VyEuwwdKbLnyXuNK4lEBjqncHAKoGfSORbq7NigP4XGe1HTSrZK9QHcQ_M0JN?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4VMTc5I3LHqam2dk2h-CsaK_jiuXPuTsiERLhKZIQTgUrr3QltcwwEF2qidGdHdYKxWxscT1rPMj6iRm_CmsmqQ_4wD_5QEsGOdyh8JqA6wAFXaPQDE8RtXxOB1wVpAqlWKmnziTDKZ-k9IW-C5KKyBp0xp7elvOXcoAu9TiOqgoVoHE8Fl2yIRr9M5TrE_z?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/pT5rc2Xe-4_ida_OYjxPfINi2Hv3A5xRQD6nMeI6CUrlhhEJF-vw5MB01tsY13_IFiyFZsba2OAYdgUZlSiFBD8fItgrnvXCVuGsBHogI9vOcMpDPQ_uZHU269Lj7Bz-4_stcWKQW3eLzPE8o9RekScTv5EvbIdAmUyqwzheLLq6jU2KobCu8cwy9pUgMoWm?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/xILRBTyEUcm9WckI7lrb1kon1-FJ810wyZQne_JucS_jT3OVSBv48zOyKXg4vVAwtGTrHpBAZCEWM7pIJ8JvZyrrxET0mtocIL2d63WSsZ3E7kFp51ixRNa3h29uXgWN7n7eV-3-TLwZsKQdnsYok49E-K4OjVtHdq2ApJBbtdaHgCsQjGrrEhG9TLCclsaq?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/pF67jbZHEUFmeU_C-O0AfBrWWNQQVCxQ12jfSrjNMasL6emLuScq8IMeiirRnKR66VOkw9tQDJMSHaAAnqsKStx3u-iMxgZ2R4IbqkdtrIlBdUkQSbQvSJWfb7p2YS1e2AKb7v2XQ8b6aGTqnALyeGX8wCdxrbUr8O5RtmSWvGLwJItSpNh1t4FSzcA10UtV?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/P-C0zh7h7UB9nC4STOCVJCF0nRf-MwhIsHfwhB2CgxmLOeR01pp2bFDcJaRXdfEC2ftD6g7-6uAU-PAB478w1yO_ogVeDcKStaBf4oxcTiDqsItef1pweSlcgBOfcsVUD2J8rUqVLiHHss-m0aeEzHy4zFutEggzOgL43aIDvlXFgLnE1Gtkh_c5Zs2fCMQI?purpose=fullsize)

* Deploy servers worldwide (PoPs = Points of Presence)
* Each edge caches content

---

# ⚙️ 5. Request Flow

```id="3k8k9o"
1. User requests content
2. DNS routes to nearest edge
3. Edge checks cache:
   → Hit → return content
   → Miss → fetch from origin
4. Cache it for future
```

---

# 📦 6. Caching Strategy

### Cache Key:

```id="l4p6l2"
URL + headers
```

### Policies:

* TTL (expiry time)
* LRU eviction
* Cache invalidation

---

# ⚡ 7. Cache Invalidation (Hard Problem 🔥)

### Methods:

* TTL expiration
* Manual purge (API)
* Versioned URLs

Example:

```id="0l5vpx"
/image_v2.png
```

---

# 🚀 8. Scaling the System

### 🔹 Problem: Global traffic

Solution:

* Multiple edge nodes
* Load balancing across regions

---

### 🔹 Problem: Hot content

Solution:

* Replicate across many edges
* Pre-warm cache

---

# 🔄 9. Load Balancing

## DNS-based routing

* Geo-based routing
* Latency-based routing

---

# ⚡ 10. Dynamic Content Handling

Not all content is cacheable.

Solution:

* Reverse proxy
* Partial caching

---

# 🔒 11. Security Features

* DDoS protection
* Rate limiting
* TLS termination (HTTPS)

---

# 📊 12. Monitoring

Track:

* Cache hit ratio
* Latency
* Error rates

---

# 🧩 13. Final Architecture

```id="4d7z0z"
        [User]
            ↓
         [DNS]
            ↓
    ┌─────────────────┐
    │   CDN Edges     │
    │  (Global PoPs)  │
    └─────────────────┘
            ↓
      [Origin Server]
```

---

# ⚖️ 14. Trade-offs

| Design Choice      | Pros        | Cons          |
| ------------------ | ----------- | ------------- |
| Aggressive caching | Fast        | Stale content |
| TTL-based          | Simple      | Less control  |
| Global replication | Low latency | High cost     |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use DNS-based geo routing to nearest edge”
* “Cache invalidation via TTL + versioning”
* “Optimize for high cache hit ratio”
* “Use CDN as reverse proxy for dynamic content”
# 🧠 1. Requirements

### ✅ Functional

* Multiple users can **edit the same document simultaneously**
* Changes appear **in real-time**
* Version history
* Cursor position of other users
* Comments (optional)

---

### ⚡ Non-Functional

* Ultra low latency (real-time typing ⚡)
* Strong consistency (no data conflicts 🚨)
* High availability
* Scalable (thousands of concurrent editors)

---

# 🏗️ 2. High-Level Architecture

```
Client → WebSocket Server → Collaboration Engine → Storage
```

### Components:

* WebSocket servers (real-time communication)
* Collaboration engine (core logic)
* Database (document storage)
* Cache (fast access)

---

# 🔑 3. Core Problem

👉 “How do multiple users edit the same document without conflicts?”

---

# 🧩 4. Two Main Approaches

## 🏆 1. Operational Transformation (OT)

Used by Google Docs

### Idea:

* Transform operations before applying

Example:

```
User A: Insert "Hello"
User B: Insert "World"
```

System transforms:

```
Final → "Hello World"
```

✔ Maintains consistency
✔ Handles concurrent edits

---

## 🆕 2. CRDT (Conflict-free Replicated Data Type)

### Idea:

* Each user has local copy
* Changes merge automatically

✔ Works offline
✔ Eventually consistent

---

## 🏆 Interview Answer:

> “I’ll use **OT** for strong real-time consistency.”

---

# ⚙️ 5. Real-Time Communication

👉 Use **WebSockets**

Why?

* Persistent connection
* Instant updates

---

# 🔄 6. Edit Flow

```
1. User types character
2. Client sends operation (insert/delete)
3. Server processes via OT
4. Broadcast to all users
5. Clients update UI
```

---

# 📦 7. Operation Format

```
{
  type: "insert",
  position: 5,
  character: "A",
  user_id: 123
}
```

---

# 🧠 8. Collaboration Engine (Core)

Responsibilities:

* Apply operations
* Transform conflicts (OT logic)
* Maintain document state

---

# 🗄️ 9. Storage Design

### Document Table

| Field   | Type |
| ------- | ---- |
| doc_id  | UUID |
| content | TEXT |
| version | INT  |

---

### Operation Log (for history)

| Field     | Type |
| --------- | ---- |
| op_id     | UUID |
| doc_id    | UUID |
| operation | JSON |

---

# ⚡ 10. Scaling the System

### 🔹 Problem: Many concurrent users

Solution:

* Partition by document ID
* Each document handled by specific server

---

### 🔹 Problem: High update rate

Solution:

* Batch operations
* Compress updates

---

# 🔄 11. Conflict Handling

Example:

```
User A deletes char at pos 3
User B inserts at pos 2
```

👉 OT transforms operations so order doesn’t break document

---

# 📊 12. Versioning

* Maintain versions of document
* Allow undo/redo

---

# 🔒 13. Security

* Authentication (JWT)
* Access control (edit/view permissions)

---

# ⚡ 14. Optimizations

* Send diffs instead of full document
* Local buffering for smoother typing
* Cursor tracking

---

# 🧩 15. Final Architecture

```
        [Client]
            ↓
     [WebSocket Server]
            ↓
 [Collaboration Engine]
        ↓        ↓
     [Cache]   [DB]
```

---

# ⚖️ 16. Trade-offs

| Approach | Pros               | Cons              |
| -------- | ------------------ | ----------------- |
| OT       | Strong consistency | Complex logic     |
| CRDT     | Offline support    | More memory usage |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use WebSockets for real-time sync”
* “Operational Transformation to resolve conflicts”
* “Partition by document ID for scaling”
* “Send operations instead of full document”


# 🧠 1. Requirements

### ✅ Functional

* Collect logs from multiple services/servers
* Support **search & filtering**
* Real-time log streaming
* Retention policies (e.g., 7–30 days)
* Alerting on patterns (errors, spikes)

---

### ⚡ Non-Functional

* High throughput (millions of logs/sec)
* Low ingestion latency ⚡
* Highly scalable & fault-tolerant
* Durable (no log loss 🚨)

---

# 🏗️ 2. High-Level Architecture

```id="g1c7hp"
Services → Log Agents → Queue → Processing → Storage → Query UI
```

---

# 🧩 3. Core Components

## 🖥️ 1. Log Producers

* Microservices, servers, apps
* Generate logs:

```id="xv2k5n"
INFO, ERROR, DEBUG
```

---

## 📡 2. Log Agents

Installed on each server:

* Collect logs
* Forward to central system

Examples (conceptually):

* Filebeat, Fluentd

---

## 📦 3. Message Queue (VERY IMPORTANT)

Use:

* Kafka / Pulsar

Why?

* Buffer spikes
* Decouple producers from consumers

```id="z6g7yq"
Logs → Queue → Consumers
```

---

## ⚙️ 4. Processing Layer

Tasks:

* Parse logs (JSON, text)
* Add metadata (timestamp, service name)
* Filter unwanted logs

---

## 🗄️ 5. Storage Layer

### 🏆 Searchable Storage

Use something like:

* Elasticsearch

Why?

* Full-text search
* Fast queries

---

### 🧊 Cold Storage

* Cheap storage (S3-like)
* For long-term logs

---

## 🔍 6. Query Layer

* UI/dashboard (like Kibana)
* Query logs:

```id="ljx3p0"
ERROR logs from service X in last 1 hour
```

---

# 🔄 7. Data Flow

```id="gph4vt"
1. Service generates logs
2. Agent collects logs
3. Push to Kafka
4. Consumers process logs
5. Store in Elasticsearch
6. User queries logs
```

---

# 🚀 8. Scaling Strategy

### 🔹 Problem: Huge log volume

Solution:

* Partition logs in Kafka
* Scale consumers horizontally

---

### 🔹 Problem: Storage size

Solution:

* Data retention policies
* Move old logs to cold storage

---

# ⚡ 9. Indexing Strategy

* Index by:

  * timestamp
  * service name
  * log level

✔ Faster queries

---

# 🔄 10. Fault Tolerance

* Replicate logs in Kafka
* Retry failed writes
* Multi-node storage cluster

---

# 🔒 11. Security

* Encrypt logs in transit
* Access control (RBAC)
* Mask sensitive data

---

# 📊 12. Monitoring & Alerts

* Detect:

  * Error spikes
  * System failures

Example:

```id="o7v6nh"
If ERROR > 1000/min → trigger alert
```

---

# 🧩 13. Final Architecture

```id="u5l9td"
 [Services]
     ↓
 [Log Agents]
     ↓
   [Kafka]
     ↓
 [Processors]
     ↓
[Elasticsearch]
     ↓
   [Dashboard]
```

---

# ⚖️ 14. Trade-offs

| Design Choice | Pros               | Cons           |
| ------------- | ------------------ | -------------- |
| Kafka queue   | Reliable buffering | Complexity     |
| Elasticsearch | Fast search        | Resource heavy |
| Cold storage  | Cost efficient     | Slow retrieval |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll decouple ingestion using Kafka”
* “Use Elasticsearch for full-text search”
* “Implement retention + cold storage strategy”
* “Partition logs for scalability”
# 🧠 1. Requirements

### ✅ Functional

* Schedule jobs:

  * One-time (run at T)
  * Recurring (cron expressions)
* Execute jobs at the correct time
* Retry on failure
* Support priorities & delays
* Allow cancellation/rescheduling

---

### ⚡ Non-Functional

* High reliability (don’t miss jobs 🚨)
* High scalability (millions of jobs)
* Accurate timing (low jitter ⚡)
* Fault-tolerant

---

# 🏗️ 2. High-Level Architecture

```id="k0l1bx"
Client → Scheduler Service → Queue → Workers → Execution
                         ↓
                       Storage
```

### Components:

* Scheduler service (brain)
* Job store (DB)
* Queue (buffer)
* Worker nodes (execute jobs)

---

# 🔑 3. Core Problem

👉 “How do we trigger millions of jobs at the right time without scanning everything?”

---

# 🧩 4. Job Storage Design

### Table: `jobs`

| Field    | Type        |
| -------- | ----------- |
| job_id   | UUID        |
| schedule | CRON / time |
| next_run | TIMESTAMP   |
| status   | ENUM        |
| payload  | JSON        |

---

# ⚙️ 5. Scheduling Strategy

## 🏆 Priority Queue (Min-Heap)

* Sort jobs by `next_run_time`

```id="3eq0nk"
Top = job with earliest execution time
```

✔ Efficient
✔ O(log n) insert

---

# ⏱️ 6. Core Scheduling Loop

```id="v4z8pq"
1. Fetch next job (earliest time)
2. If time reached → push to queue
3. Update next_run (for recurring jobs)
4. Repeat
```

---

# 📦 7. Queue Layer

Use:

* Kafka / RabbitMQ / SQS

Why?

* Decouple scheduling from execution
* Handle spikes

---

# 👷 8. Worker Layer

Workers:

* Consume jobs from queue
* Execute tasks

```id="2z2v7c"
Send email / process payment / run script
```

---

# 🔄 9. Recurring Jobs (CRON)

Example:

```id="b7f2e2"
0 9 * * * → run every day at 9 AM
```

Flow:

* After execution → compute next run
* Update DB

---

# 🚀 10. Scaling the System

### 🔹 Problem: Millions of jobs

Solution:

* Partition jobs by time buckets
* Use sharded schedulers

---

### 🔹 Problem: Scheduler bottleneck

Solution:

* Multiple scheduler instances
* Leader election (only one active per partition)

---

# ⚡ 11. Time Bucketing Optimization

Instead of scanning all jobs:

```id="u8v2l9"
Bucket jobs into:
10:00–10:01
10:01–10:02
```

✔ Fast lookup
✔ Reduced load

---

# 🔒 12. Fault Tolerance

* Store jobs in durable DB
* Use retries with backoff
* Track job state

---

# 🔁 13. Idempotency (VERY IMPORTANT)

👉 Ensure job runs **only once**

Solution:

* Unique job IDs
* Deduplication logic

---

# ⚡ 14. Retry Strategy

```id="t9x5y8"
Retry after:
1 min → 5 min → 15 min
```

* Move failed jobs to DLQ

---

# 🧩 15. Final Architecture

```id="1u5x5k"
        [Client]
            ↓
   [Scheduler Service]
            ↓
        [Job DB]
            ↓
         [Queue]
            ↓
        [Workers]
```

---

# ⚖️ 16. Trade-offs

| Design Choice     | Pros     | Cons         |
| ----------------- | -------- | ------------ |
| Central scheduler | Simple   | Bottleneck   |
| Distributed       | Scalable | Complex      |
| Queue-based       | Reliable | Slight delay |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use a priority queue (min-heap) for scheduling”
* “Decouple execution using message queues”
* “Use time bucketing to avoid full scans”
* “Ensure idempotency to avoid duplicate execution”
# 🧠 1. Requirements

### ✅ Functional

* Recommend movies/shows to users
* Personalized feed (“Top picks for you”)
* Trending / popular content
* Similar items (“Because you watched X”)

---

### ⚡ Non-Functional

* Low latency (results <100ms ⚡)
* Highly scalable (millions of users)
* Accurate recommendations
* Near real-time updates

---

# 🏗️ 2. High-Level Architecture

```id="u7x4k2"
Client → API Gateway → Recommendation Service
                          ↓
     ┌───────────────┬───────────────┐
 [User Data]   [Content Data]   [ML Models]
```

---

# 🔑 3. Core Idea

👉 “Recommend content based on **user behavior + similarity + trends**”

---

# 🧩 4. Types of Recommendation

## 🏆 1. Collaborative Filtering

* Users with similar behavior → similar recommendations

Example:

```id="3i9p0p"
User A likes X, Y  
User B likes X → recommend Y
```

✔ Most powerful

---

## 🎯 2. Content-Based Filtering

* Recommend similar content

Example:

```id="xk8f7q"
Watched "Action movie" → suggest similar action movies
```

---

## 🔥 3. Hybrid Approach (Best)

👉 Combine both
👉 Used by Netflix

---

# 📊 5. Data Collection

Collect:

* Watch history
* Likes/ratings
* Search queries
* Time spent

---

# ⚙️ 6. ML Pipeline

### Offline Processing (Heavy computation)

```id="n3h5g2"
1. Collect data
2. Train models
3. Generate recommendations
4. Store results
```

---

### Online Serving (Fast)

```id="j3k9q1"
User request → Fetch precomputed recommendations
```

---

# 🗄️ 7. Storage Design

### User Profile Store

```id="g4t8y6"
user_id → preferences, history
```

### Recommendation Store

```id="p7m2v9"
user_id → [movie1, movie2, movie3]
```

---

# ⚡ 8. Real-Time Updates

👉 For freshness:

* Update recommendations on:

  * New watch
  * New content

Use:

* Streaming pipelines (Kafka)

---

# 🚀 9. Scaling Strategy

### 🔹 Problem: Heavy ML computation

Solution:

* Offline batch processing
* Precompute recommendations

---

### 🔹 Problem: Low latency

Solution:

* Cache recommendations (Redis)
* Serve from memory

---

# ⚡ 10. Ranking Layer (VERY IMPORTANT)

Final ranking:

```id="d8j6z3"
score = w1*similarity + w2*popularity + w3*recency
```

👉 Re-rank results before showing

---

# 🔄 11. Cold Start Problem

### New User:

* Show trending content

### New Content:

* Use content-based filtering

---

# 🔒 12. Privacy

* Handle user data securely
* Anonymize sensitive info

---

# 🧩 13. Final Architecture

```id="c6v8n2"
        [Client]
            ↓
     [API Gateway]
            ↓
 [Recommendation Service]
        ↓         ↓
   [Cache]     [DB]
        ↓
   [ML Pipeline]
```

---

# ⚖️ 14. Trade-offs

| Approach                | Pros         | Cons               |
| ----------------------- | ------------ | ------------------ |
| Collaborative filtering | Accurate     | Cold start problem |
| Content-based           | Simple       | Limited diversity  |
| Hybrid                  | Best results | Complex            |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use hybrid recommendation system”
* “Precompute recommendations offline”
* “Use caching for low latency”
* “Handle cold start with trending content”
# 🧠 1. Requirements

### ✅ Functional

* Detect fraudulent transactions in real-time
* Flag or block suspicious activity
* Support rules + ML-based detection
* Generate alerts for review

---

### ⚡ Non-Functional

* Ultra low latency (**<100 ms ⚡**)
* High accuracy (low false positives 🚨)
* Highly scalable (millions of transactions/day)
* Reliable & auditable

---

# 🏗️ 2. High-Level Architecture

```id="y2g7zn"
Transaction → Fraud Service → Decision Engine → Approve / Block
                         ↓
                    Feature Store + Models
```

---

# 🔑 3. Core Idea

👉 “Score every transaction using **rules + ML models** and take action instantly”

---

# 🧩 4. Detection Approaches

## 🧾 1. Rule-Based System

Examples:

```id="8b3y6t"
- Transaction > ₹50,000 → flag
- Different country → flag
- Too many transactions in short time → flag
```

✔ Fast
❌ Limited intelligence

---

## 🤖 2. Machine Learning Model

* Trained on historical data
* Predicts fraud probability

✔ Detects complex patterns
❌ Needs training & tuning

---

## 🏆 Best Approach: Hybrid

👉 Combine rules + ML

---

# ⚙️ 5. Real-Time Processing Flow

```id="z3f7r9"
1. User makes transaction
2. Fetch user history (feature store)
3. Apply rules
4. Run ML model → risk score
5. Decision:
   → Approve
   → Flag
   → Block
```

---

# 📦 6. Feature Engineering (VERY IMPORTANT)

Features used:

* Transaction amount
* Location (geo distance)
* Device/IP
* Time of transaction
* User behavior patterns

---

# 🗄️ 7. Feature Store

```id="d7x4l1"
user_id → recent transactions, behavior
```

👉 Stored in fast DB (Redis / NoSQL)

---

# ⚡ 8. Decision Engine

Example:

```id="u4m9t2"
score < 0.3 → approve  
0.3–0.7 → flag  
> 0.7 → block
```

---

# 🚀 9. Scaling Strategy

### 🔹 Problem: High transaction rate

Solution:

* Stream processing (Kafka)
* Stateless fraud services

---

### 🔹 Problem: Low latency

Solution:

* Precompute features
* In-memory storage

---

# 🔄 10. Model Training Pipeline

Offline:

```id="k6n2y5"
1. Collect historical data
2. Train model
3. Validate accuracy
4. Deploy model
```

---

# ⚡ 11. Real-Time Updates

* Update user behavior continuously
* Adapt to new fraud patterns

---

# 🔒 12. Security & Compliance

* Encryption of sensitive data
* Audit logs for decisions
* Regulatory compliance (banking rules)

---

# 📊 13. Monitoring

Track:

* Fraud detection rate
* False positives
* Model drift

---

# 🧩 14. Final Architecture

```id="f9n3c8"
      [Transaction]
            ↓
     [Fraud Service]
        ↓        ↓
 [Rules Engine] [ML Model]
        ↓        ↓
     [Decision Engine]
            ↓
    Approve / Block
```

---

# ⚖️ 15. Trade-offs

| Approach | Pros         | Cons               |
| -------- | ------------ | ------------------ |
| Rules    | Fast, simple | Easy to bypass     |
| ML       | Accurate     | Complex            |
| Hybrid   | Best balance | Harder to maintain |

---

# 🔥 Interview Power Moves

Say these and you’ll stand out:

* “I’ll use hybrid detection (rules + ML)”
* “Precompute features for low latency”
* “Use streaming pipeline for real-time scoring”
* “Continuously retrain models to handle new fraud patterns”
