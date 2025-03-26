import os
import sys
import re
import pathlib
import numpy as np
import torch
import torchaudio
import logging
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import random
import time
from typing import Optional, List
from typing import Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor
import torchaudio.transforms as T
'''

原始的

'''
# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..' % os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import


class VoiceRecognitionManager:
    def __init__(self, model_id: str, comparison_dir: str, local_model_dir: str = 'pretrained'):
        self.model_id = model_id
        self.comparison_dir = comparison_dir
        self.local_model_dir = local_model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.counter_file = os.path.join(self.comparison_dir, "counter.txt")
        self.embedding_cache_dir = os.path.join(self.comparison_dir, "embedding_cache")

        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        self.keep_latest_n = 1  # 仅保留最新 N 个文件
        self.counter = self._load_counter()

        self._initialize_model()


    def _initialize_model(self):
        """加载预训练模型和特征提取器"""
        conf = supports[self.model_id]
        model_path = pathlib.Path(self.local_model_dir) / self.model_id.split('/')[1] / conf['model_pt']
        model_path.parent.mkdir(exist_ok=True, parents=True)

        self.model = dynamic_import(conf['model']['obj'])(**conf['model']['args'])
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device).eval()

        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def _load_counter(self) -> int:
        """加载命名计数器"""
        return int(open(self.counter_file).read().strip()) if os.path.exists(self.counter_file) else 2

    def _save_counter(self, counter: int):
        """保存命名计数器"""
        with open(self.counter_file, "w") as f:
            f.write(str(counter))

    @staticmethod
    def _generate_name(counter: int) -> str:
        """生成新用户名"""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        name = ""
        while counter > 0:
            counter -= 1
            name = alphabet[counter % 26] + name
            counter //= 26
        return name

    def _load_wav(self, wav_file: str, obj_fs: int = 16000):
        """加载音频文件"""
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            wav = torchaudio.transforms.Resample(fs, obj_fs)(wav)
        return wav[0].unsqueeze(0) if wav.shape[0] > 1 else wav


# ####################################################################
#     def _load_wav(self, wav_file: str, obj_fs: int = 16000):
#         """加载音频文件，并进行数据增强（拼接三次）"""
#         wav, fs = torchaudio.load(wav_file)

#         if fs != obj_fs:
#             wav = torchaudio.transforms.Resample(fs, obj_fs)(wav)

#         # 确保音频为单通道
#         if wav.shape[0] > 1:
#             wav = wav.mean(dim=0, keepdim=True)

#         # 进行数据增强：音频自身拼接三次
#         wav = torch.cat([wav, wav, wav], dim=1)  

#         return wav
#####################################################################



    def _compute_embedding(self, wav_file: str) -> np.ndarray:
        """计算音频嵌入"""
        wav = self._load_wav(wav_file)
        feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(feat).cpu().numpy().squeeze(0)

    def _get_cache_path(self, user: str, filename: str = None) -> str:
        """获取缓存路径"""
        user_cache_dir = os.path.join(self.embedding_cache_dir, user)
        return os.path.join(user_cache_dir, os.path.splitext(filename)[0] + '.npy') if filename else user_cache_dir

    def _save_embedding_cache(self, user: str, filename: str, embedding: np.ndarray):
        """保存嵌入缓存"""
        os.makedirs(self._get_cache_path(user), exist_ok=True)
        np.save(self._get_cache_path(user, filename), embedding)


    def _load_embedding_cache(self, user: str, filename: str) -> Optional[np.ndarray]:
        """加载嵌入缓存"""
        cache_path = self._get_cache_path(user, filename)
        logger.info(f"===========================loading: {cache_path}================================")
        if os.path.exists(cache_path):
            try:
                embedding = np.load(cache_path)

                logger.info(f"===========================found: {cache_path}================================")
                return embedding
            except Exception as e:
                logger.error(f"Error loading cache for {user}/{filename}: {e}")
                return None
        else:
            logger.info(f"===========================not found: {cache_path}================================")
        return None



    def _compute_user_embeddings(self, user: str, wav_files: List[str], force_recompute: bool = False) -> Optional[np.ndarray]:
        if not wav_files:
            logger.warning(f"No wav files found for user {user}")
            return None
        embeddings = []
        for wav_file in wav_files:
            if not force_recompute:
                # 尝试加载缓存
                cached_embedding = self._load_embedding_cache(user, wav_file)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue
            # 计算新的嵌入向量
            wav_path = os.path.join(self.comparison_dir, user, wav_file)
            try:
                embedding = self._compute_embedding(wav_path)
                embeddings.append(embedding)
                # 保存缓存
                self._save_embedding_cache(user, wav_file, embedding)
            except Exception as e:
                logger.error(f"Error computing embedding for {user}/{wav_file}: {e}")
                return None
        if not embeddings:
            logger.warning(f"Failed to compute any embeddings for user {user}")
            return None
        return np.stack(embeddings)


    def _update_embeddings_cache_after_delete(self, user: str, deleted_files: List[str]) -> None:
        """删除文件后清理对应的缓存文件
        Args:
            user: 用户名
            deleted_files: 被删除的音频文件列表
        """
        for filename in deleted_files:
            cache_path = self._get_cache_path(user, filename)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    logger.error(f"Error deleting cache file for {user}/{filename}: {e}")


    def _get_non_latest_files(self, user_path: str) -> List[str]:
        """获取用户目录下除了最新的N个文件之外的所有音频文件
        Args:
            user_path: 用户目录路径
        Returns:
            可以参与合并的音频文件列表
        """
        wav_files = [f for f in os.listdir(user_path) if f.endswith('.wav')]
        if not wav_files:
            return []
        # 获取文件创建时间
        wav_files_with_time = []
        for wav_file in wav_files:
            file_path = os.path.join(user_path, wav_file)
            creation_time = os.path.getctime(file_path)
            wav_files_with_time.append((wav_file, creation_time))
        # 按创建时间排序
        wav_files_with_time.sort(key=lambda x: x[1])
        # 返回除了最新的N个文件之外的所有文件
        return [f[0] for f in wav_files_with_time[:-self.keep_latest_n]]


    def merge_user(self, similarity_threshold: float = 0.5, max_workers: int = 4) -> None:
        """合并相似用户"""
        logger.info("Starting user similarity analysis and merging process...")
        
        # 1. 获取所有用户及其样本，排除最新的文件
        user_dirs = [d for d in os.listdir(self.comparison_dir) 
                    if os.path.isdir(os.path.join(self.comparison_dir, d)) and d != "embedding_cache"]
        
        if len(user_dirs) <= 1:
            logger.info("No need to merge: insufficient users")
            return

        # 2. 加载或计算所有用户的嵌入向量（排除最新的文件）
        user_embeddings = {}
        for user in user_dirs:
            user_path = os.path.join(self.comparison_dir, user)
            # 获取非最新的文件
            wav_files = self._get_non_latest_files(user_path)
            
            if wav_files:  # 只在有可用文件时处理
                embeddings = self._compute_user_embeddings(user, wav_files)
                if embeddings is not None:
                    user_embeddings[user] = embeddings

        # 3. 计算用户间的相似度矩阵
        def compute_pair_similarity(pair: Tuple[str, str]) -> Tuple[Tuple[str, str], float]:
            user1, user2 = pair
            try:
                embeddings1 = torch.from_numpy(user_embeddings[user1]).to(self.device)
                embeddings2 = torch.from_numpy(user_embeddings[user2]).to(self.device)
                
                # 计算交叉相似度矩阵
                similarity = torch.nn.CosineSimilarity(dim=-1)(
                    embeddings1.unsqueeze(1),
                    embeddings2.unsqueeze(0)
                )
                
                # 计算相似度得分（保持原有逻辑）
                n1 = len(embeddings1)
                n2 = len(embeddings2)
                
                if n1 <= 3 and n2 <= 3:
                    final_sim = torch.max(torch.cat([similarity.max(dim=1).values, 
                                                   similarity.max(dim=0).values])).item()
                else:
                    k1 = min(5, n1)
                    k2 = min(5, n2)
                    topk_mean_1 = torch.topk(similarity.max(dim=1).values, k=k1).values.mean()
                    topk_mean_2 = torch.topk(similarity.max(dim=0).values, k=k2).values.mean()
                    
                    mean_sim = similarity.mean()
                    std_sim = similarity.std()
                    
                    n1_tensor = torch.tensor(n1, dtype=torch.float32, device=self.device)
                    n2_tensor = torch.tensor(n2, dtype=torch.float32, device=self.device)
                    weight1 = 1.0 - torch.exp(-n1_tensor / 5.0)
                    weight2 = 1.0 - torch.exp(-n2_tensor / 5.0)
                    conf_weight = 1.0 / (1.0 + std_sim)
                    
                    final_sim = (
                        0.5 * (topk_mean_1 * weight1 + topk_mean_2 * weight2) / (weight1 + weight2) +
                        0.2 * mean_sim * conf_weight +
                        0.3 * torch.max(similarity.max(dim=1).values.mean(), 
                                      similarity.max(dim=0).values.mean())
                    ).item()
                
                return (user1, user2), final_sim
            except Exception as e:
                logger.error(f"Error computing similarity for {user1} and {user2}: {e}")
                return (user1, user2), -1

        # 生成用户对并计算相似度
        user_pairs = [(u1, u2) for i, u1 in enumerate(user_embeddings.keys())
                     for u2 in list(user_embeddings.keys())[i+1:]]
        
        similarity_matrix = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(compute_pair_similarity, pair)
                      for pair in user_pairs]
            for future in futures:
                pair, sim = future.result()
                if sim >= 0:
                    similarity_matrix[pair] = sim

        # 4. 基于相似度进行聚类
        clusters = []
        used_users = set()

        # 获取每个用户的音频数量
        user_audio_counts = {}
        for user in user_embeddings.keys():
            user_path = os.path.join(self.comparison_dir, user)
            wav_files = [f for f in os.listdir(user_path) if f.endswith('.wav')]
            user_audio_counts[user] = len(wav_files)

        for user1 in user_embeddings.keys():
            if user1 in used_users:
                continue

            if user_audio_counts[user1] <= 3:
                continue

            current_cluster = {user1}
            used_users.add(user1)

            for user2 in user_embeddings.keys():
                if user2 not in used_users:
                    if user_audio_counts[user2] <= 3:
                        continue
                        
                    pair = tuple(sorted([user1, user2]))
                    if pair in similarity_matrix and similarity_matrix[pair] >= similarity_threshold:
                        current_cluster.add(user2)
                        used_users.add(user2)

            if len(current_cluster) > 1:
                clusters.append(current_cluster)

        # 5. 合并聚类的用户
        def merge_cluster(cluster: set) -> None:
            try:
                main_user = min(cluster)
                main_user_path = os.path.join(self.comparison_dir, main_user)
                
                for other_user in cluster - {main_user}:
                    other_user_path = os.path.join(self.comparison_dir, other_user)
                    other_wav_files = [f for f in os.listdir(other_user_path) if f.endswith('.wav')]
                    
                    # 转移每个音频文件并更新缓存
                    for wav_file in other_wav_files:
                        # 生成新的基于时间戳的文件名
                        new_wav_file = self._generate_timestamp_filename(main_user)
                        src_path = os.path.join(other_user_path, wav_file)
                        dst_path = os.path.join(main_user_path, new_wav_file)
                        
                        # 复制文件
                        shutil.copy2(src_path, dst_path)
                        
                        # 加载或重新计算嵌入向量
                        embedding = self._load_embedding_cache(other_user, wav_file)
                        if embedding is None:
                            embedding = self._compute_embedding(src_path)
                        
                        # 保存新的缓存
                        self._save_embedding_cache(main_user, new_wav_file, embedding)
                    
                    # 删除其他用户的目录和缓存目录
                    shutil.rmtree(other_user_path)
                    other_cache_dir = self._get_cache_path(other_user)
                    if os.path.exists(other_cache_dir):
                        shutil.rmtree(other_cache_dir)
                
            except Exception as e:
                logger.error(f"Error merging cluster {cluster}: {e}")

        # 并行合并聚类
        logger.info("Merging clustered users...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(merge_cluster, clusters))

        logger.info(f"Merge complete. {len(clusters)} clusters merged.")
        for i, cluster in enumerate(clusters, 1):
            logger.info(f"Cluster {i}: {cluster}")

    def make_collection_more_sim(self, max_workers: int = 4) -> None:
        """清理每个用户目录下的不相似音频，使得集合更加相似
        
        Args:
            max_workers: 并行计算的最大线程数，默认4
        """
        logger.info("Starting to optimize user audio collections...")
        
        # 1. 获取所有用户目录
        user_dirs = [d for d in os.listdir(self.comparison_dir) 
                    if os.path.isdir(os.path.join(self.comparison_dir, d)) and d != "embedding_cache"]
        
        def process_user(user: str) -> None:
            """处理单个用户的音频集合"""
            try:
                user_path = os.path.join(self.comparison_dir, user)
                # 获取非最新的文件
                wav_files = self._get_non_latest_files(user_path)
                
                # 如果音频数量少于6，跳过处理
                if len(wav_files) < 6:
                    logger.info(f"Skipping user {user}: insufficient samples ({len(wav_files)} < 6)")
                    return
                
                # 获取文件创建时间
                wav_files_with_time = []
                for wav_file in wav_files:
                    file_path = os.path.join(user_path, wav_file)
                    creation_time = os.path.getctime(file_path)
                    wav_files_with_time.append((wav_file, creation_time))
                
                # 按创建时间排序
                wav_files_with_time.sort(key=lambda x: x[1])
                latest_wav = wav_files_with_time[-1][0]
                
                # 获取或计算嵌入向量
                embeddings_array = self._compute_user_embeddings(user, wav_files)
                if embeddings_array is None:
                    logger.warning(f"Skipping user {user}: failed to compute embeddings")
                    return

                # 计算相似度矩阵
                embeddings_tensor = torch.from_numpy(embeddings_array).to(self.device)
                similarity_matrix = torch.nn.CosineSimilarity(dim=-1)(
                    embeddings_tensor.unsqueeze(1),
                    embeddings_tensor.unsqueeze(0)
                )
                
                # 计算每个音频与其他音频的平均相似度
                mean_similarities = []
                for i, wav_file in enumerate(wav_files):
                    # 跳过最新的文件
                    if wav_file == latest_wav:
                        continue
                        
                    # 排除自身的相似度（对角线上的1）
                    other_similarities = torch.cat([similarity_matrix[i,:i], similarity_matrix[i,i+1:]])
                    mean_sim = other_similarities.mean().item()
                    mean_similarities.append((mean_sim, wav_file))
                
                # 如果剩余文件数量太少，跳过处理
                if len(mean_similarities) < 5:
                    logger.info(f"Skipping user {user}: insufficient samples after excluding latest file")
                    return
                
                # 按平均相似度排序
                mean_similarities.sort(key=lambda x: x[0])
                
                # 删除平均相似度最低的10%的音频文件
                num_to_remove = max(1, int(len(mean_similarities) * 0.1))
                files_to_remove = mean_similarities[:num_to_remove]
                
                # 删除文件和对应的缓存
                for mean_sim, wav_file in files_to_remove:
                    file_path = os.path.join(user_path, wav_file)
                    os.remove(file_path)
                    # 删除对应的缓存文件
                    self._update_embeddings_cache_after_delete(user, [wav_file])
                    logger.info(f"Removed {wav_file} from user {user} (avg similarity: {mean_sim:.4f})")
                
                logger.info(f"Processed user {user}: removed {num_to_remove} files out of {len(wav_files)} (kept latest file: {latest_wav})")
                
            except Exception as e:
                logger.error(f"Error processing user {user}: {e}")
        
        # 并行处理所有用户
        logger.info(f"Processing {len(user_dirs)} users...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(process_user, user_dirs))
        
        logger.info("Collection optimization complete.")




    def _save_similarity_analysis(self, input_wav: str, results: List[Tuple[float, str]], best_score: float, best_user: str) -> None:
        """保存相似度分析结果到文件
        Args:
            input_wav: 输入音频文件路径
            results: 所有有效的相似度计算结果
            best_score: 最佳匹配分数
            best_user: 最佳匹配用户
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        analysis_dir = os.path.join(self.comparison_dir, "similarity_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        analysis_file = os.path.join(analysis_dir, f"similarity_analysis_{timestamp}.log")
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("Similarity Analysis Results:\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入输入音频信息
            f.write(f"Input Audio: {input_wav}\n")
            f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model ID: {self.model_id}\n\n")
            
            # 写入最佳匹配结果
            f.write("Best Match:\n")
            f.write(f"  - User: {best_user}\n")
            f.write(f"  - Score: {best_score:.4f}\n\n")
            
            # 写入所有有效结果
            f.write("All Valid Results:\n")
            # 按相似度分数降序排序
            sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
            for score, user in sorted_results:
                f.write(f"  - User: {user}, Score: {score:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
        
        logger.info(f"Similarity analysis results saved to: {analysis_file}")

    def process_audio(self, input_wav: str, threshold: float, max_workers: int = 4, timeout: int = 30) -> Tuple[float, str]:
        """处理音频，计算相似度，决定是否新增用户或增加用户音频
        
        Args:
            input_wav: 输入音频文件路径
            threshold: 相似度阈值
            max_workers: 最大线程数
            timeout: 超时时间（秒）
        """

        logger.info("==========================begin to merge users============================")
        self.merge_user()
        logger.info("==========================end to merge users============================")

        logger.info("==========================begin to make_collection_more_sim users============================")
        self.make_collection_more_sim()
        logger.info("==========================end to make_collection_more_sim users============================")

        target_embedding = self._compute_embedding(input_wav)
        user_dirs = [d for d in os.listdir(self.comparison_dir) 
                    if os.path.isdir(os.path.join(self.comparison_dir, d)) and d != "embedding_cache"]

        if not user_dirs:
            return self._create_new_user(input_wav)

        def compare_with_user(user: str) -> Tuple[float, str]:
            try:
                user_path = os.path.join(self.comparison_dir, user)
                wav_files = [f for f in os.listdir(user_path) if f.endswith('.wav')]
                
                if not wav_files:
                    return -1, user

                # 使用缓存系统批量获取嵌入向量
                embeddings = []
                for wav_file in wav_files:
                    # 先尝试从缓存加载
                    cached_embedding = self._load_embedding_cache(user, wav_file)
                    if cached_embedding is not None:
                        embeddings.append(cached_embedding)
                    else:
                        # 如果缓存不存在，计算并保存
                        wav_path = os.path.join(user_path, wav_file)
                        try:
                            embedding = self._compute_embedding(wav_path)
                            self._save_embedding_cache(user, wav_file, embedding)
                            embeddings.append(embedding)
                        except Exception as e:
                            logger.error(f"Error computing embedding for {user}/{wav_file}: {e}")
                            # Skip this file and continue with the next one
                            continue

                if not embeddings:
                    return -1, user
                
                # 批量计算相似度
                embeddings_tensor = torch.from_numpy(np.stack(embeddings)).to(self.device)
                target_tensor = torch.from_numpy(target_embedding).unsqueeze(0).to(self.device)
                
                # 计算相似度矩阵
                similarity = torch.nn.CosineSimilarity(dim=-1)(
                    embeddings_tensor,
                    target_tensor.expand(embeddings_tensor.shape[0], -1)
                )
                
                # 根据样本数量选择不同的计算策略
                n_samples = len(similarity)
                
                if n_samples < 100:  # 小样本情况
                    # 1. 计算最大相似度和平均相似度
                    max_sim = similarity.max()
                    mean_sim = similarity.mean()
                    
                    # 2. 小样本时使用简单加权策略
                    if n_samples == 1:
                        # 单样本情况：直接使用该样本的相似度
                        final_score = similarity.item()
                    else:
                        # 2-3个样本情况：最大相似度占更大权重
                        final_score = (0.7 * max_sim + 0.3 * mean_sim).item()
                    
                    logger.info(f"============================{user}==================================")
                    logger.info(f"User: {user}, Similarity Scores (Small Sample Mode):")
                    logger.info(f"- Samples: {n_samples}")
                    logger.info(f"- Max: {max_sim:.4f}")
                    logger.info(f"- Mean: {mean_sim:.4f}")
                    logger.info(f"- Final Score: {final_score:.4f}")
                    
                else:  # 样本数量充足的情况
                    # 1. 计算统计特征
                    max_sim = similarity.max()
                    mean_sim = similarity.mean()
                    std_sim = similarity.std()
                    
                    # 2. 计算Top-K平均
                    k = min(5, n_samples)
                    topk_mean = torch.topk(similarity, k=k).values.mean()
                    
                    # 3. 计算可靠性权重
                    n = torch.tensor(n_samples, dtype=torch.float32, device=self.device)
                    sample_weight = 1.0 - torch.exp(-n / 10.0)  # 样本数量权重
                    conf_weight = 1.0 / (1.0 + std_sim)  # 一致性权重
                    
                    # 4. 计算加权最终分数
                    final_score = (
                        0.4 * topk_mean * sample_weight +  # Top-K 相似度
                        0.3 * mean_sim * conf_weight +    # 加权平均相似度
                        0.3 * max_sim                     # 最大相似度
                ).item()
                    
                    logger.info(f"============================{user}==================================")
                    logger.info(f"User: {user}, Similarity Scores (Normal Mode):")
                    logger.info(f"- Top-{k} mean: {topk_mean:.4f}")
                    logger.info(f"- Mean: {mean_sim:.4f} (std: {std_sim:.4f})")
                    logger.info(f"- Max: {max_sim:.4f}")
                    logger.info(f"- Final Score: {final_score:.4f}")
                
                return final_score, user
            except Exception as e:
                logging.error(f"Error processing user {user}: {str(e)}")
                return -1, user

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(compare_with_user, user) for user in user_dirs]
                results = [future.result(timeout=timeout) for future in futures]

            # 过滤掉错误的结果（得分为-1的）
            valid_results = [r for r in results if r[0] > -1]
            
            if not valid_results:
                logging.warning("No valid comparison results found")
                return self._create_new_user(input_wav)

            best_score, best_user = max(valid_results, key=lambda x: x[0])
            
            # 保存相似度分析结果
            self._save_similarity_analysis(input_wav, valid_results, best_score, best_user)

            if best_score >= threshold:
                return self._add_audio_to_existing_user(input_wav, best_user, best_score)
            else:
                return self._create_new_user(input_wav)
                
        except TimeoutError:
            logging.error("Processing timeout")
            raise
        except Exception as e:
            logging.error(f"Error in process_audio: {str(e)}")
            raise

    def _generate_timestamp_filename(self, user_name: str) -> str:
        """生成基于毫秒时间戳的文件名
        Args:
            user_name: 用户名
        Returns:
            格式为 {user_name}_{timestamp}.wav 的文件名
        """
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        return f"{user_name}_{timestamp}.wav"

    def _create_new_user(self, input_wav: str) -> Tuple[float, str]:
        """创建新用户"""
        user_name = self._generate_name(self.counter)
        user_dir = os.path.join(self.comparison_dir, user_name)
        os.makedirs(user_dir, exist_ok=True)

        new_wav_file = self._generate_timestamp_filename(user_name)
        new_audio_path = os.path.join(user_dir, new_wav_file)
        shutil.copyfile(input_wav, new_audio_path)
        
        # 计算并保存新文件的缓存
        new_embedding = self._compute_embedding(new_audio_path)
        self._save_embedding_cache(user_name, new_wav_file, new_embedding)
        self._save_counter(self.counter + 1)

        return 0.0, user_name


    def _add_audio_to_existing_user(self, input_wav: str, user_name: str, score: float) -> Tuple[float, str]:
        """给现有用户添加音频（最多存 1000 个）"""
        user_dir = os.path.join(self.comparison_dir, user_name)
        user_wavs = [f for f in os.listdir(user_dir) if f.endswith('.wav')]

        # 按照文件的创建时间排序，找到最旧的文件
        user_wavs_with_paths = [(f, os.path.join(user_dir, f)) for f in user_wavs]
        user_wavs_with_paths.sort(key=lambda x: os.path.getctime(x[1]))

        if len(user_wavs) >= 1000:
            # 如果文件数量达到 1000，删除最旧的文件
            old_wav_file = user_wavs_with_paths[0][0]
            old_audio_path = user_wavs_with_paths[0][1]
            os.remove(old_audio_path)
            # 删除旧文件的缓存
            self._update_embeddings_cache_after_delete(user_name, [old_wav_file])

        # 生成新的时间戳文件名
        new_wav_file = self._generate_timestamp_filename(user_name)
        new_audio_path = os.path.join(user_dir, new_wav_file)

        # 复制新文件
        shutil.copyfile(input_wav, new_audio_path)
        
        # 计算新文件的嵌入向量并保存缓存
        new_embedding = self._compute_embedding(new_audio_path)
        self._save_embedding_cache(user_name, new_wav_file, new_embedding)

        return score, user_name
    

    

ERes2NetV2_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}


supports = {
    # ERes2Net trained on 200k labeled speakers
    'iic/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    # ERes2NetV2 trained on 200k labeled speakers
    'iic/speech_eres2netv2_sv_zh-cn_16k-common': {
        'revision': 'v1.0.1', 
        'model': ERes2NetV2_COMMON,
        'model_pt': 'pretrained_eres2netv2.ckpt',
    },

}
