import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

#  CONFIG ##############################################################################################################
FOV_DEG = 360               # LiDAR field of view
NUM_RAYS = 720              # angular resolution
MAX_RANGE = 12.0            # LiDAR max distance (m)
RANGE_NOISE_STD = 0.02      # m
DROPOUT_PROB = 0.01         # random miss rate

# Clustering params (Euclidean region-growing a.k.a. DBSCAN-like)
EPS = 0.35                  # neighborhood radius (m)
MIN_PTS = 8                 # minimum points to form a cluster

# Obstacles: some rectangles & circles in the scene
OBSTACLES = [
    ("rect", (-6.0, -4.0,  2.0,  4.5)),     # (xmin,xmax,ymin,ymax)
    ("rect", ( 3.8,  5.8,  1.0,  2.3)),
    ("rect", (-1.0,  1.0, -3.0, -2.0)),
    ("circle", ( 5.5, -3.0, 1.0)),          # (cx,cy,r)
    ("circle", (-4.0, -2.0, 0.9)),
]

CAR_POSE = np.array([0.0, 0.0, 0.0])  # (x, y, yaw) — car at origin, facing +x


#  GEOMETRY UTILS ##############################################################################################
def rotate(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return R @ v

def ray_circle_intersection(o, d, circle):
    cx, cy, r = circle
    oc = o - np.array([cx, cy])
    b = 2 * np.dot(d, oc)
    c = np.dot(oc, oc) - r*r
    disc = b*b - 4*c
    if disc < 0:
        return np.inf
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    ts = [t for t in (t1, t2) if t >= 0]
    return min(ts) if ts else np.inf

def ray_segment_intersection(o, d, p, q):
    v = q - p
    # Solve o + t d = p + u v  ->  [d, -v] [t; u] = p - o
    A = np.array([[d[0], -v[0]],[d[1], -v[1]]])
    b = p - o
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < 1e-9:
        return np.inf
    invA = (1.0/det) * np.array([[ A[1,1], -A[0,1]],[-A[1,0], A[0,0]]])
    t, u = invA @ b
    if t >= 0 and 0 <= u <= 1:
        return t
    return np.inf

def rect_edges(rect):
    xmin, xmax, ymin, ymax = rect
    p1, p2, p3, p4 = np.array([xmin,ymin]), np.array([xmax,ymin]), np.array([xmax,ymax]), np.array([xmin,ymax])
    return [(p1,p2),(p2,p3),(p3,p4),(p4,p1)]


#  LIDAR SIM ##############################################################################################################
def simulate_lidar(pose, obstacles, fov_deg, num_rays, max_range, noise_std, dropout):
    x, y, yaw = pose
    o = np.array([x, y])
    thetas = np.linspace(-np.deg2rad(fov_deg)/2.0, np.deg2rad(fov_deg)/2.0, num_rays) + yaw
    
    ranges = np.full(num_rays, max_range)
    
    for i, th in enumerate(thetas):
        d = np.array([np.cos(th), np.sin(th)])
        t_min = np.inf
        
        for typ, params in obstacles:
            if typ == "circle":
                t = ray_circle_intersection(o, d, params)
                if t < t_min:
                    t_min = t
            else:
                for p, q in rect_edges(params):
                    t = ray_segment_intersection(o, d, p, q)
                    if t < t_min:
                        t_min = t
        
        r = min(t_min, max_range)
        # Noise + dropout
        if np.random.rand() < dropout:
            r = max_range
        else:
            r = np.clip(r + np.random.randn()*noise_std, 0.0, max_range)
        
        ranges[i] = r
    
    # Convert to point cloud in world frame
    xs = o[0] + ranges * np.cos(thetas)
    ys = o[1] + ranges * np.sin(thetas)
    # Only keep hits strictly less than max_range (i.e., not infinite/sky)
    mask = ranges < max_range - 1e-6
    pc = np.stack([xs[mask], ys[mask]], axis=1)
    return pc, thetas, ranges


#  CLUSTERING (Region-Growing) ##############################################################################################
def euclidean_cluster(points, eps=0.4, min_pts=5):
    if len(points) == 0:
        return np.array([], dtype=int)
    N = len(points)
    labels = -np.ones(N, dtype=int)
    visited = np.zeros(N, dtype=bool)
    
    # Precompute squared distances matrix (could be large; OK for a class demo)
    d2 = np.sum(points**2, axis=1, keepdims=True) + np.sum(points**2, axis=1) - 2*points@points.T
    
    cluster_id = 0
    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(d2[i] <= eps*eps)[0]
        if len(neighbors) < min_pts:
            labels[i] = -1  # noise
            continue
        # start a new cluster
        labels[i] = cluster_id
        seeds = list(neighbors[neighbors != i])
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                nbh = np.where(d2[j] <= eps*eps)[0]
                if len(nbh) >= min_pts:
                    # extend
                    for n in nbh:
                        if labels[n] == -1:
                            labels[n] = cluster_id  # border becomes core
                        if not visited[n]:
                            seeds.append(int(n))
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] == -2:  # unassigned (we used -1 only; keep for clarity)
                labels[j] = cluster_id
        cluster_id += 1
    return labels


#  RUN PIPELINE ##############################################################################################################
pc, thetas, ranges = simulate_lidar(
    CAR_POSE, OBSTACLES, FOV_DEG, NUM_RAYS, MAX_RANGE, RANGE_NOISE_STD, DROPOUT_PROB
)
labels = euclidean_cluster(pc, eps=EPS, min_pts=MIN_PTS)

# Compute simple cluster stats (centroids & AABBs)
cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
clusters = []
for cid in cluster_ids:
    pts = pc[labels == cid]
    ctr = pts.mean(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    clusters.append((cid, ctr, mins, maxs, len(pts)))




#  VISUALIZATION #####################################################################################################################
# 1) original scene + LiDAR rays
plt.figure(figsize=(8, 8))
plt.title("2D LiDAR Simulation")
# Draw obstacles
for typ, p in OBSTACLES:
    if typ == "rect":
        xmin, xmax, ymin, ymax = p
        xs = [xmin, xmax, xmax, xmin, xmin]
        ys = [ymin, ymin, ymax, ymax, ymin]
        plt.plot(xs, ys)
    else:
        cx, cy, r = p
        ang = np.linspace(0, 2*np.pi, 200)
        plt.plot(cx + r*np.cos(ang), cy + r*np.sin(ang))
# Car and heading
plt.plot(CAR_POSE[0], CAR_POSE[1], marker="s")
h = np.array([np.cos(CAR_POSE[2]), np.sin(CAR_POSE[2])])
plt.arrow(CAR_POSE[0], CAR_POSE[1], 0.8*h[0], 0.8*h[1], head_width=0.2, length_includes_head=True)
# LiDAR rays
for th, r in zip(thetas, ranges):
    if r < MAX_RANGE - 1e-6:
        plt.plot([CAR_POSE[0], CAR_POSE[0] + r*np.cos(th)], [CAR_POSE[1], CAR_POSE[1] + r*np.sin(th)], color="orange", alpha=0.3)
# Point cloud
plt.scatter(pc[:,0], pc[:,1], s=10, color="red")
plt.axis("equal")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()


# 2) RESULT
plt.figure(figsize=(8, 8))
plt.title("2D LiDAR Point Cloud & Euclidean Clustering")
# Car
plt.plot(CAR_POSE[0], CAR_POSE[1], marker="s")
# Heading arrow
h = np.array([np.cos(CAR_POSE[2]), np.sin(CAR_POSE[2])])
plt.arrow(CAR_POSE[0], CAR_POSE[1], 0.8*h[0], 0.8*h[1], head_width=0.2, length_includes_head=True)

# Point cloud colored by cluster labels (default colormap)
plt.scatter(pc[:,0], pc[:,1], c=labels, s=10)

# Draw cluster
for cid, ctr, mn, mx, sz in clusters:
    # AABB
    plt.plot([mn[0], mx[0], mx[0], mn[0], mn[0]], [mn[1], mn[1], mx[1], mx[1], mn[1]])

plt.axis("equal")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

