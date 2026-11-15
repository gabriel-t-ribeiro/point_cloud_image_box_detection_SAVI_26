import os
import numpy as np
import open3d as o3d

# Parametros (ajustar a escala das nuvens)
FOLDER = "point_clouds/table"
VOXEL_SIZE = 0.03                    # downsampling para o FPFH e o RANSAC
RANSAC_MAX_ITERATIONS = 50000        # numero maximo de iteracoes do RANSAC
RANSAC_CONFIDENCE = 0.999            # o RANSAC pára quando atinge esta confianca
NORMAL_RADIUS = VOXEL_SIZE * 2       # vizinhanca para estimar as normais
FPFH_RADIUS = VOXEL_SIZE * 5         # vizinhanca para calcular as features FPFH
RANSAC_DISTANCE = VOXEL_SIZE * 1.5   # distancia maxima entre pontos associados no RANSAC
ICP_THRESHOLD = 0.05                 # distancia maxima de associacao no ICP
WEIGHT_SCALE = 1.5                   # distancia a origem (m) a que o peso de um ponto desce para 0.5

registration = o3d.pipelines.registration


def draw(clouds, window_name):
    # Mostra as nuvens rodadas 180 graus em x, para ter a mesa virada para cima (so afeta a visualizacao, nao os dados)
    flip_x = np.diag([1.0, -1.0, -1.0, 1.0])
    rotated = []
    for cloud in clouds:
        cloud_rotated = o3d.geometry.PointCloud(cloud)
        cloud_rotated.transform(flip_x)
        rotated.append(cloud_rotated)
    o3d.visualization.draw_geometries(rotated, window_name=window_name)


def preprocess(point_cloud):
    # Downsampling + normais + features FPFH de cada ponto
    down = point_cloud.voxel_down_sample(VOXEL_SIZE)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30)
    )
    fpfh = registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100)
    )
    return down, fpfh


def match_features(source_fpfh, target_fpfh):
    # Associacoes entre features: pares (i, j) em que i e j sao o vizinho mais proximo um do outro
    source_tree = o3d.geometry.KDTreeFlann(source_fpfh)
    target_tree = o3d.geometry.KDTreeFlann(target_fpfh)
    source_to_target = [target_tree.search_knn_vector_xd(source_fpfh.data[:, i], 1)[1][0]
                        for i in range(source_fpfh.data.shape[1])]
    target_to_source = [source_tree.search_knn_vector_xd(target_fpfh.data[:, j], 1)[1][0]
                        for j in range(target_fpfh.data.shape[1])]
    return np.array([(i, j) for i, j in enumerate(source_to_target) if target_to_source[j] == i])


def estimate_rigid_transform(src, dst):
    # Transformacao rigida (Kabsch) que leva os pontos src para os pontos dst
    src_center, dst_center = src.mean(axis=0), dst.mean(axis=0)
    U, _, Vt = np.linalg.svd((src - src_center).T @ (dst - dst_center))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dst_center - R @ src_center
    return T


def ransac(source_down, target_down, source_fpfh, target_fpfh):
    # RANSAC em que as associacoes perto da origem sao mais escolhidas e valem mais na pontuacao
    rng = np.random.default_rng(0)

    pairs = match_features(source_fpfh, target_fpfh)
    if len(pairs) < 3:
        raise ValueError("Poucas associacoes entre features para correr o RANSAC")

    src = np.asarray(source_down.points)[pairs[:, 0]]
    dst = np.asarray(target_down.points)[pairs[:, 1]]
    src_normals = np.asarray(source_down.normals)[pairs[:, 0]]
    dst_normals = np.asarray(target_down.normals)[pairs[:, 1]]

    # Ajustar os pesos com base na distâsncia à origem (priveligiar ter a mesa alinhada em vez do backgropund)
    weights = 1.0 / (1.0 + (np.linalg.norm(dst, axis=1) / WEIGHT_SCALE) ** 2)
    probabilities = weights / weights.sum()

    best_T = np.eye(4)
    best_score = 0.0
    max_iterations = RANSAC_MAX_ITERATIONS
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Escolher 3 associacoes, com mais probabilidade para as perto da origem
        idx = rng.choice(len(pairs), 3, replace=False, p=probabilities)
        a, b = src[idx], dst[idx]

        # Verificacao do comprimento das arestas (como CorrespondenceCheckerBasedOnEdgeLength(0.9))
        src_edges = np.linalg.norm(a[[0, 0, 1]] - a[[1, 2, 2]], axis=1)
        dst_edges = np.linalg.norm(b[[0, 0, 1]] - b[[1, 2, 2]], axis=1)
        if np.any(src_edges < 0.9 * dst_edges) or np.any(dst_edges < 0.9 * src_edges):
            continue

        T = estimate_rigid_transform(a, b)

        # Verificacao das normais
        rotated_normals = src_normals[idx] @ T[:3, :3].T
        if np.any(np.sum(rotated_normals * dst_normals[idx], axis=1) < np.cos(0.52)):
            continue

        # Pontuacao: soma dos pesos das associacoes que ficam a menos de RANSAC_DISTANCE
        moved = src @ T[:3, :3].T + T[:3, 3]
        inliers = np.linalg.norm(moved - dst, axis=1) < RANSAC_DISTANCE
        score = weights[inliers].sum()

        if score > best_score:
            best_score, best_T = score, T

            # Numero de iteracoes necessario para a confianca pedida
            ratio = inliers.mean()
            if ratio < 1.0:
                needed = np.log(1.0 - RANSAC_CONFIDENCE) / np.log(1.0 - ratio ** 3)
            else:
                needed = 0
            max_iterations = min(RANSAC_MAX_ITERATIONS, needed)

    return best_T


def align_cloud(source, target):
    # Devolve a transformacao que alinha source com target (RANSAC + ICP)
    source_down, source_fpfh = preprocess(source)
    target_down, target_fpfh = preprocess(target)

    # RANSAC com pesos: estimativa inicial a partir das associacoes entre features
    T_ransac = ransac(source_down, target_down, source_fpfh, target_fpfh)
    ransac_result = registration.evaluate_registration(source_down, target_down, RANSAC_DISTANCE, T_ransac)
    print(f"  RANSAC -> fitness: {ransac_result.fitness:.3f} | rmse: {ransac_result.inlier_rmse:.4f}")

    # Normais nas nuvens originais (necessarias para o ICP ponto a plano)
    for point_cloud in (source, target):
        point_cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30)
        )

    # ICP ponto a plano nas nuvens originais, a partir do resultado do RANSAC
    icp_result = registration.registration_icp(
        source, target, ICP_THRESHOLD, T_ransac,
        registration.TransformationEstimationPointToPlane(),
        registration.ICPConvergenceCriteria(max_iteration=100),
    )
    print(f"  ICP    -> fitness: {icp_result.fitness:.3f} | rmse: {icp_result.inlier_rmse:.4f}")
    return icp_result.transformation


def main():
    # Resultados do RANSAC repetiveis entre execucoes
    # o3d.utility.random.seed(0)

    # Carregar todas as nuvens da pasta (ordem alfabetica)
    file_names = sorted(name for name in os.listdir(FOLDER) if name.lower().endswith(".ply"))
    if len(file_names) == 0:
        raise FileNotFoundError(f"No point cloud file (.ply) found in {FOLDER}")
    clouds = [o3d.io.read_point_cloud(os.path.join(FOLDER, name)) for name in file_names]

    # Uma cor diferente para cada nuvem. É APLICADO À NUVEM MESMO, NÃO À COPIA
    for i, cloud in enumerate(clouds):
        hue = i / len(clouds)
        cloud.paint_uniform_color(0.5 + 0.4 * np.cos(2 * np.pi * (hue + np.array([0.0, 1 / 3, 2 / 3]))))

    # Visualizacao 1: nuvens na orientacao original
    # draw(clouds, "Orientacao original")

    # A primeira nuvem e a referencia; cada nova nuvem e alinhada com todas as anteriores
    aligned_clouds = [o3d.geometry.PointCloud(clouds[0])]

    for i in range(1, len(clouds)):
        print(f"\nA alinhar {file_names[i]}")

        # Target: todas as nuvens ja alinhadas, juntas numa so
        target = o3d.geometry.PointCloud()
        for cloud in aligned_clouds:
            target += cloud

        transformation = align_cloud(clouds[i], target)

        aligned = o3d.geometry.PointCloud(clouds[i])
        aligned.transform(transformation)
        aligned_clouds.append(aligned)

    # Visualizacao 2: todas as nuvens alinhadas
    draw(aligned_clouds, "Nuvens alinhadas")


if __name__ == "__main__":
    main()