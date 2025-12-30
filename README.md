
import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# ===ПАРАМЕТРЫ МОДЕЛИРОВАНИЯ ====
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Время моделирования
SIM_TIME = 600

# Профиль нагрузки (время начала, время окончания, интенсивность запросов/сек)
REQUEST_RATE_LOW = 5
REQUEST_RATE_HIGH = 50
LOAD_PROFILE = [
    (0, 150, REQUEST_RATE_LOW),
    (150, 300, REQUEST_RATE_HIGH),
    (300, 450, REQUEST_RATE_LOW),
    (450, 600, REQUEST_RATE_HIGH)
]

# Параметры инстансов
REQUEST_PROCESS_TIME = 0.5  # среднее время обработки запроса
INSTANCE_CAPACITY = 10      # максимальное количество одновременных запросов на инстанс
FIXED_NUM_INSTANCES = 5     # фиксированное количество инстансов

# ==== КЛАСС ВИРТУАЛЬНОЙ МАШИНЫ ====
class CloudInstance:
    """Класс, представляющий виртуальную машину (инстанс) в облаке"""
    
    def __init__(self, env, instance_id: int, capacity: int):
        self.env = env
        self.instance_id = instance_id
        self.capacity = capacity
        self.resource = simpy.Resource(env, capacity=capacity)
        self.active_requests = 0
        self.metrics_log = []
    
    def process_request(self, request_id: int):
        """Обработка одного запроса"""
        self.active_requests += 1
        self._log_metrics()
        
        # Имитация времени обработки (экспоненциальное распределение)
        processing_time = random.expovariate(1.0 / REQUEST_PROCESS_TIME)
        yield self.env.timeout(processing_time)
        
        self.active_requests -= 1
        self._log_metrics()
    
    def _log_metrics(self):
        """Логирование метрик инстанса"""
        utilization = self.active_requests / self.capacity
        self.metrics_log.append({
            'time': self.env.now,
            'instance_id': self.instance_id,
            'active_requests': self.active_requests,
            'queue_size': len(self.resource.queue),
            'utilization': utilization
        })

# === ФУНКЦИИ МОДЕЛИРОВАНИЯ ====
def run_fixed_instances_simulation():
    """
    Основная функция для запуска моделирования с фиксированными инстансами
    """
    env = simpy.Environment()
    
    # Создание фиксированного пула инстансов
    instances = [CloudInstance(env, i, INSTANCE_CAPACITY) 
                 for i in range(FIXED_NUM_INSTANCES)]
    
    # Логи для сбора результатов
    request_log = []
    system_metrics = []
    
    def request_generator():
        """Генератор входящих запросов"""
        request_id = 0
        
        for start_time, end_time, rate in LOAD_PROFILE:
            while env.now < end_time:
                if env.now >= start_time:
                    # Создание нового запроса
                    arrival_time = env.now
                    request_id += 1
                    
                    # Выбор инстанса с наименьшей нагрузкой
                    target_instance = min(
                        instances,
                        key=lambda x: x.active_requests + len(x.resource.queue)
                    )
                    
                    # Запуск обработки запроса
                    env.process(
                        handle_request(request_id, arrival_time, target_instance)
                    )
                    
                    # Логирование системных метрик (раз в 0.1 секунды)
                    if env.now % 0.1 < 0.01:
                        total_active = sum(i.active_requests for i in instances)
                        total_queue = sum(len(i.resource.queue) for i in instances)
                        avg_utilization = np.mean([
                            i.active_requests / i.capacity for i in instances
                        ])
                        
                        system_metrics.append({
                            'time': env.now,
                            'total_active_requests': total_active,
                            'total_queue_size': total_queue,
                            'avg_utilization': avg_utilization,
                            'active_instances': FIXED_NUM_INSTANCES
                        })
                
                # Генерация следующего запроса
                interval = random.expovariate(rate)
                yield env.timeout(interval)
    
    def handle_request(request_id: int, arrival_time: float, instance: CloudInstance):
        """Обработка одного запроса"""
        with instance.resource.request() as req:
            # Ожидание в очереди инстанса
            queue_start_time = env.now
            yield req
            queue_time = env.now - queue_start_time
            
            # Обработка запроса
            start_time = env.now
            yield env.process(instance.process_request(request_id))
            processing_time = env.now - start_time
            
            # Расчет общего времени отклика
            response_time = env.now - arrival_time
            
            # Логирование результатов
            request_log.append({
                'request_id': request_id,
                'arrival_time': arrival_time,
                'start_time': start_time,
                'end_time': env.now,
                'response_time': response_time,
                'queue_time': queue_time,
                'processing_time': processing_time,
                'instance_id': instance.instance_id,
                'status': 'processed'
            })
    
    # Запуск процессов
    env.process(request_generator())
    env.run(until=SIM_TIME)
    
    return pd.DataFrame(request_log), pd.DataFrame(system_metrics), instances

# === ОСНОВНАЯ ПРОГРАММА ====
if __name__ == "__main__":
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ СЦЕНАРИЯ 1: СИСТЕМА БЕЗ АВТОМАСШТАБИРОВАНИЯ")
    print("=" * 70)
    
    # Запуск моделирования
    print("\nЗапуск моделирования...")
    request_df, metrics_df, instances = run_fixed_instances_simulation()
    
    # Вывод статистики
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ:")
    print("=" * 70)
    
    print(f"\nОбщая статистика:")
    print(f"• Время моделирования: {SIM_TIME} сек")
    print(f"• Общее количество запросов: {len(request_df)}")
    print(f"• Фиксированное количество инстансов: {FIXED_NUM_INSTANCES}")
    print(f"• Пропускная способность инстанса: {INSTANCE_CAPACITY} запр/сек")
    
    print(f"\nМетрики производительности:")
    print(f"• Среднее время отклика: {request_df['response_time'].mean():.3f} сек")
    print(f"• Максимальное время отклика: {request_df['response_time'].max():.3f} сек")
    print(f"• 95-й процентиль времени отклика: "
          f"{np.percentile(request_df['response_time'], 95):.3f} сек")
    print(f"• Среднее время в очереди: {request_df['queue_time'].mean():.3f} сек")
    print(f"• Среднее время обработки: {request_df['processing_time'].mean():.3f} сек")
    
    print(f"\nМетрики использования ресурсов:")
    print(f"• Средняя утилизация инстансов: "
          f"{metrics_df['avg_utilization'].mean() * 100:.1f}%")
    print(f"• Максимальная утилизация: "
          f"{metrics_df['avg_utilization'].max() * 100:.1f}%")
    print(f"• Средний размер очереди: {metrics_df['total_queue_size'].mean():.1f} запросов")
    print(f"• Максимальный размер очереди: {metrics_df['total_queue_size'].max():.0f} запросов")
    
    # Анализ по периодам нагрузки
    print(f"\nАнализ по периодам нагрузки:")
    
    for i, (start, end, rate) in enumerate(LOAD_PROFILE):
        period_requests = request_df[
            (request_df['arrival_time'] >= start) & 
            (request_df['arrival_time'] < end)
        ]
        
        load_type = "ВЫСОКАЯ" if rate == REQUEST_RATE_HIGH else "НИЗКАЯ"
        print(f"\nПериод {i+1} ({load_type} нагрузка, {start}-{end} сек):")
        print(f"  • Количество запросов: {len(period_requests)}")
        if len(period_requests) > 0:
            print(f"  • Среднее время отклика: {period_requests['response_time'].mean():.3f} сек")
            print(f"  • Максимальное время отклика: {period_requests['response_time'].max():.3f} сек")
    
    # Сохранение результатов
    print(f"\n" + "=" * 70)
    request_df.to_csv('scenario1_requests.csv', index=False)
    metrics_df.to_csv('scenario1_metrics.csv', index=False)
    print("Результаты сохранены в файлы:")
    print("• scenario1_requests.csv - детальная информация о запросах")
    print("• scenario1_metrics.csv - системные метрики")
    
    print(f"\nМоделирование завершено успешно!") 

import simpy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# ==== ПАРАМЕТРЫ МОДЕЛИРОВАНИЯ ====
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Время моделирования
SIM_TIME = 600

# Профиль нагрузки
REQUEST_RATE_LOW = 5
REQUEST_RATE_HIGH = 50
LOAD_PROFILE = [
    (0, 150, REQUEST_RATE_LOW),
    (150, 300, REQUEST_RATE_HIGH),
    (300, 450, REQUEST_RATE_LOW),
    (450, 600, REQUEST_RATE_HIGH)
]

# Параметры инстансов
REQUEST_PROCESS_TIME = 0.5
INSTANCE_CAPACITY = 10

# Параметры автомасштабирования
MIN_INSTANCES = 2
MAX_INSTANCES = 10
TARGET_UTILIZATION = 0.7  # 70%
COOLDOWN_PERIOD = 30  # сек
SCALE_OUT_THRESHOLD = 1.1  # 110% от целевой утилизации
SCALE_IN_THRESHOLD = 0.9   # 90% от целевой утилизации
INSTANCE_STARTUP_DELAY = 20  # время запуска нового инстанса
MONITORING_INTERVAL = 5  # интервал мониторинга метрик

# === ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ ====
class InstanceStatus(Enum):
    """Статусы виртуальной машины"""
    STARTING = "starting"
    RUNNING = "running"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

@dataclass
class ScalingDecision:
    """Решение о масштабировании"""
    action: str  # 'scale_out', 'scale_in', 'none'
    timestamp: float
    current_size: int
    target_size: int
    avg_utilization: float

# ==== КЛАСС ВИРТУАЛЬНОЙ МАШИНЫ ===
class CloudInstance:
    """Класс, представляющий виртуальную машину (инстанс) в облаке"""
    
    def __init__(self, env, instance_id: int, capacity: int):
        self.env = env
        self.instance_id = instance_id
        self.capacity = capacity
        self.resource = simpy.Resource(env, capacity=capacity)
        self.status = InstanceStatus.STARTING
        self.start_time = env.now
        self.end_time = None
        self.active_requests = 0
        self.metrics_log = []
    
    def process_request(self, request_id: int):
        """Обработка одного запроса"""
        if self.status != InstanceStatus.RUNNING:
            return
        
        self.active_requests += 1
        self._log_metrics()
        
        # Имитация времени обработки
        processing_time = random.expovariate(1.0 / REQUEST_PROCESS_TIME)
        yield self.env.timeout(processing_time)
        
        self.active_requests -= 1
        self._log_metrics()
    
    def mark_for_termination(self):
        """Пометить инстанс для удаления"""
        if self.status == InstanceStatus.RUNNING:
            self.status = InstanceStatus.TERMINATING
    
    def terminate(self):
        """Завершить работу инстанса"""
        self.status = InstanceStatus.TERMINATED
        self.end_time = self.env.now
    
    def _log_metrics(self):
        """Логирование метрик инстанса"""
        utilization = self.active_requests / self.capacity
        self.metrics_log.append({
            'time': self.env.now,
            'instance_id': self.instance_id,
            'status': self.status.value,
            'active_requests': self.active_requests,
            'queue_size': len(self.resource.queue),
            'utilization': utilization
        })

# === КЛАСС АВТОСКЕЙЛЕРА ===
class Autoscaler:
    """Класс, реализующий логику автомасштабирования"""
    
    def __init__(self, env, instance_group, params: Dict):
        self.env = env
        self.instance_group = instance_group
        self.params = params
        
        # Состояние автоскейлера
        self.last_scaling_time = 0
        self.scaling_decisions = []
        self.metrics_history = []
    
    def monitor(self):
        """Основной цикл мониторинга и принятия решений"""
        while True:
            # Ждем следующий интервал мониторинга
            yield self.env.timeout(self.params['monitoring_interval'])
            
            # Рассчитываем текущую утилизацию
            current_utilization = self._calculate_utilization()
            current_size = len(self.instance_group.get_running_instances())
            
            # Логируем метрики
            self.metrics_history.append({
                'time': self.env.now,
                'group_size': current_size,
                'avg_utilization': current_utilization
            })
            
            # Проверяем, можно ли выполнять масштабирование
            if self.env.now - self.last_scaling_time < self.params['cooldown_period']:
                continue
            
            # Принимаем решение о масштабировании
            decision = self._make_scaling_decision(current_utilization, current_size)
            
            if decision.action != 'none':
                self.last_scaling_time = self.env.now
                self.scaling_decisions.append(decision)
                
                if decision.action == 'scale_out':
                    yield env.process(self._scale_out(decision.target_size - current_size))
                elif decision.action == 'scale_in':
                    yield env.process(self._scale_in(current_size - decision.target_size))
    
    def _calculate_utilization(self) -> float:
        """Расчет средней утилизации группы инстансов"""
        running_instances = self.instance_group.get_running_instances()
        
        if not running_instances:
            return 0.0
        
        utilizations = []
        for instance in running_instances:
            if instance.metrics_log:
                # Берем последнюю запись метрик
                last_metric = instance.metrics_log[-1]
                utilizations.append(last_metric['utilization'])
        
        return np.mean(utilizations) if utilizations else 0.0
    
    def _make_scaling_decision(self, utilization: float, current_size: int) -> ScalingDecision:
        """Принятие решения о масштабировании"""
        target_util = self.params['target_utilization']
        min_instances = self.params['min_instances']
        max_instances = self.params['max_instances']
        
        # Проверка условий для scale-out
        if (utilization > target_util * self.params['scale_out_threshold'] and 
            current_size < max_instances):
            # Рассчитываем необходимое количество инстансов
            needed_instances = int(np.ceil(current_size * utilization / target_util))
            target_size = min(needed_instances, max_instances)
            
            return ScalingDecision(
                action='scale_out',
                timestamp=self.env.now,
                current_size=current_size,
                target_size=target_size,
                avg_utilization=utilization
            )
        
        # Проверка условий для scale-in
        elif (utilization < target_util * self.params['scale_in_threshold'] and 
              current_size > min_instances):
            # Рассчитываем оптимальное количество инстансов
            optimal_instances = int(np.floor(current_size * utilization / target_util))
            target_size = max(optimal_instances, min_instances)
            
            return ScalingDecision(
                action='scale_in',
                timestamp=self.env.now,
                current_size=current_size,
                target_size=target_size,
                avg_utilization=utilization
            )
        
        # Без изменений
        return ScalingDecision(
            action='none',
            timestamp=self.env.now,
            current_size=current_size,
            target_size=current_size,
            avg_utilization=utilization
        )
    
    def _scale_out(self, num_to_add: int):
        """Добавление новых инстансов"""
        for i in range(num_to_add):
            # Задержка на запуск инстанса
            yield self.env.timeout(self.params['instance_startup_delay'])
            
            # Создание нового инстанса
            new_instance = self.instance_group.create_instance()
            
            # Запуск инстанса
            new_instance.status = InstanceStatus.RUNNING
            
            print(f"[{self.env.now:.1f}] SCALE-OUT: Запущен новый инстанс #{new_instance.instance_id}")
    
    def _scale_in(self, num_to_remove: int):
        """Удаление инстансов"""
        running_instances = self.instance_group.get_running_instances()
        
        if not running_instances:
            return
        
        # Выбираем наименее загруженные инстансы для удаления
        instances_to_remove = sorted(
            running_instances,
            key=lambda x: x.active_requests
        )[:num_to_remove]
        
        for instance in instances_to_remove:
            instance.mark_for_termination()
            print(f"[{self.env.now:.1f}] SCALE-IN: Инстанс #{instance.instance_id} помечен для удаления")
            
            # Ждем завершения обработки текущих запросов
            while instance.active_requests > 0:
                yield self.env.timeout(1.0)
            
            # Завершаем работу инстанса
            instance.terminate()
            self.instance_group.remove_instance(instance)
            
            print(f"[{self.env.now:.1f}] SCALE-IN: Инстанс #{instance.instance_id} удален")

# === КЛАСС ГРУППЫ ИНСТАНСОВ ===
class InstanceGroup:
    """Управление группой виртуальных машин"""
    
    def __init__(self, env, initial_size: int, capacity: int):
        self.env = env
        self.capacity = capacity
        self.instances = []
        self.next_instance_id = 0
        
        # Создаем начальные инстансы
        for i in range(initial_size):
            self.create_instance()
    
    def create_instance(self) -> CloudInstance:
        """Создание нового инстанса"""
        instance = CloudInstance(self.env, self.next_instance_id, self.capacity)
        self.instances.append(instance)
        self.next_instance_id += 1
        return instance
    
    def remove_instance(self, instance: CloudInstance):
        """Удаление инстанса из группы"""
        if instance in self.instances:
            self.instances.remove(instance)
    
    def get_running_instances(self) -> List[CloudInstance]:
        """Получение списка работающих инстансов"""
        return [i for i in self.instances 
                if i.status in [InstanceStatus.RUNNING, InstanceStatus.TERMINATING]]
    
    def get_available_instances(self) -> List[CloudInstance]:
        """Получение списка доступных для обработки запросов инстансов"""
        return [i for i in self.instances 
                if i.status == InstanceStatus.RUNNING]

# === ОСНОВНАЯ ФУНКЦИЯ МОДЕЛИРОВАНИЯ ===
def run_autoscaling_simulation():
    """Запуск моделирования с автомасштабированием"""
    env = simpy.Environment()
    
    # Создание группы инстансов
    instance_group = InstanceGroup(env, MIN_INSTANCES, INSTANCE_CAPACITY)
    
    # Параметры автоскейлера
    autoscaler_params = {
        'min_instances': MIN_INSTANCES,
        'max_instances': MAX_INSTANCES,
        'target_utilization': TARGET_UTILIZATION,
        'cooldown_period': COOLDOWN_PERIOD,
        'scale_out_threshold': SCALE_OUT_THRESHOLD,
        'scale_in_threshold': SCALE_IN_THRESHOLD,
        'instance_startup_delay': INSTANCE_STARTUP_DELAY,
        'monitoring_interval': MONITORING_INTERVAL
    }
    
    # Создание автоскейлера
    autoscaler = Autoscaler(env, instance_group, autoscaler_params)
    
    # Логи для сбора результатов
    request_log = []
    system_metrics = []
    
    def request_generator():
        """Генератор входящих запросов"""
        request_id = 0
        
        for start_time, end_time, rate in LOAD_PROFILE:
            while env.now < end_time:
                if env.now >= start_time:
                    # Создание нового запроса
                    arrival_time = env.now
                    request_id += 1
                    
                    # Выбор доступного инстанса
                    available_instances = instance_group.get_available_instances()
                    
                    if available_instances:
                        # Выбираем наименее загруженный инстанс
                        target_instance = min(
                            available_instances,
                            key=lambda x: x.active_requests + len(x.resource.queue)
                        )
                        
                        # Запуск обработки запроса
                        env.process(
                            handle_request(request_id, arrival_time, target_instance)
                        )
                    else:
                        # Нет доступных инстансов - запрос теряется
                        request_log.append({
                            'request_id': request_id,
                            'arrival_time': arrival_time,
                            'end_time': env.now,
                            'response_time': 0,
                            'instance_id': -1,
                            'status': 'rejected'
                        })
                    
                    # Логирование системных метрик
                    if env.now % 0.1 < 0.01:
                        running_instances = instance_group.get_running_instances()
                        total_active = sum(i.active_requests for i in running_instances)
                        total_queue = sum(len(i.resource.queue) for i in running_instances)
                        
                        avg_utilization = 0.0
                        if running_instances:
                            utilizations = []
                            for inst in running_instances:
                                if inst.metrics_log:
                                    last_metric = inst.metrics_log[-1]
                                    utilizations.append(last_metric['utilization'])
                            if utilizations:
                                avg_utilization = np.mean(utilizations)
                        
                        system_metrics.append({
                            'time': env.now,
                            'total_active_requests': total_active,
                            'total_queue_size': total_queue,
                            'avg_utilization': avg_utilization,
                            'active_instances': len(running_instances)
                        })
                
                # Генерация следующего запроса
                interval = random.expovariate(rate)
                yield env.timeout(interval)
    
    def handle_request(request_id: int, arrival_time: float, instance: CloudInstance):
        """Обработка одного запроса"""
        with instance.resource.request() as req:
            # Ожидание в очереди инстанса
            queue_start_time = env.now
            yield req
            queue_time = env.now - queue_start_time
            
            # Проверка, не помечен ли инстанс для удаления
            if instance.status == InstanceStatus.TERMINATING:
                request_log.append({
                    'request_id': request_id,
                    'arrival_time': arrival_time,
                    'end_time': env.now,
                    'response_time': env.now - arrival_time,
                    'instance_id': instance.instance_id,
                    'status': 'terminated_during_processing'
                })
                return
            
            # Обработка запроса
            start_time = env.now
            yield env.process(instance.process_request(request_id))
            processing_time = env.now - start_time
            
            # Расчет общего времени отклика
            response_time = env.now - arrival_time
            
            # Логирование результатов
            request_log.append({
                'request_id': request_id,
                'arrival_time': arrival_time,
                'start_time': start_time,
                'end_time': env.now,
                'response_time': response_time,
                'queue_time': queue_time,
                'processing_time': processing_time,
                'instance_id': instance.instance_id,
                'status': 'processed'
            })
    
    # Запуск процессов
    env.process(request_generator())
    env.process(autoscaler.monitor())
    env.run(until=SIM_TIME)
    
    return (
        pd.DataFrame(request_log),
        pd.DataFrame(system_metrics),
        instance_group,
        autoscaler
    )

# === ОСНОВНАЯ ПРОГРАММА ===
if __name__ == "__main__":
    print("=" * 70)
    print("МОДЕЛИРОВАНИЕ СЦЕНАРИЯ 2: СИСТЕМА С АВТОМАСШТАБИРОВАНИЕМ")
    print("=" * 70)
    
    # Запуск моделирования
    print("\nЗапуск моделирования...")
    request_df, metrics_df, instance_group, autoscaler = run_autoscaling_simulation()
    
    # Вывод статистики
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ:")
    print("=" * 70)
    
    print(f"\nОбщая статистика:")
    print(f"• Время моделирования: {SIM_TIME} сек")
    print(f"• Общее количество запросов: {len(request_df)}")
    print(f"• Начальное количество инстансов: {MIN_INSTANCES}")
    print(f"• Максимальное количество инстансов: {MAX_INSTANCES}")
    print(f"• Целевая утилизация: {TARGET_UTILIZATION*100}%")
    
    # Анализ запросов
    processed_requests = request_df[request_df['status'] == 'processed']
    rejected_requests = request_df[request_df['status'] == 'rejected']
    terminated_requests = request_df[request_df['status'] == 'terminated_during_processing']
    
    print(f"\nМетрики производительности:")
    print(f"• Успешно обработано: {len(processed_requests)} запросов "
          f"({len(processed_requests)/len(request_df)*100:.1f}%)")
    print(f"• Отклонено: {len(rejected_requests)} запросов "
          f"({len(rejected_requests)/len(request_df)*100:.1f}%)")
    print(f"• Прервано при масштабировании: {len(terminated_requests)} запросов")
    
    if len(processed_requests) > 0:
        print(f"• Среднее время отклика: {processed_requests['response_time'].mean():.3f} сек")
        print(f"• Максимальное время отклика: {processed_requests['response_time'].max():.3f} сек")
        print(f"• 95-й процентиль времени отклика: "
              f"{np.percentile(processed_requests['response_time'], 95):.3f} сек")
        print(f"• Среднее время в очереди: {processed_requests['queue_time'].mean():.3f} сек")
        print(f"• Среднее время обработки: {processed_requests['processing_time'].mean():.3f} сек")
    
    print(f"\nМетрики использования ресурсов:")
    print(f"• Средняя утилизация инстансов: "
          f"{metrics_df['avg_utilization'].mean() * 100:.1f}%")
    print(f"• Средний размер группы инстансов: {metrics_df['active_instances'].mean():.1f}")
    print(f"• Минимальный размер группы: {metrics_df['active_instances'].min()}")
    print(f"• Максимальный размер группы: {metrics_df['active_instances'].max()}")
    
    # Анализ операций масштабирования
    print(f"\nОперации масштабирования:")
    scale_out_count = sum(1 for d in autoscaler.scaling_decisions if d.action == 'scale_out')
    scale_in_count = sum(1 for d in autoscaler.scaling_decisions if d.action == 'scale_in')
    
    print(f"• Всего операций scale-out: {scale_out_count}")
    print(f"• Всего операций scale-in: {scale_in_count}")
    print(f"• Всего операций масштабирования: {len(autoscaler.scaling_decisions)}")
    
    # Расчет стоимости
    total_instance_hours = 0
    for instance in instance_group.instances:
        if instance.end_time is not None:
            runtime = instance.end_time - instance.start_time
        else:
            runtime = SIM_TIME - instance.start_time
        total_instance_hours += runtime / 3600
    
    estimated_cost = total_instance_hours * 0.05  # примерная стоимость $0.05 за час
    
    print(f"\nЭкономические показатели:")
    print(f"• Суммарное время работы инстансов: {total_instance_hours:.2f} ч")
    print(f"• Оценочная стоимость: ${estimated_cost:.2f}")
    
    # Анализ по периодам нагрузки
    print(f"\nАнализ по периодам нагрузки:")
    
    for i, (start, end, rate) in enumerate(LOAD_PROFILE):
        period_requests = processed_requests[
            (processed_requests['arrival_time'] >= start) & 
            (processed_requests['arrival_time'] < end)
        ]
        
        load_type = "ВЫСОКАЯ" if rate == REQUEST_RATE_HIGH else "НИЗКАЯ"
        print(f"\nПериод {i+1} ({load_type} нагрузка, {start}-{end} сек):")
        print(f"  • Количество обработанных запросов: {len(period_requests)}")
        if len(period_requests) > 0:
            print(f"  • Среднее время отклика: {period_requests['response_time'].mean():.3f} сек")
            
            # Находим средний размер группы в этот период
            period_metrics = metrics_df[
                (metrics_df['time'] >= start) & 
                (metrics_df['time'] < end)
            ]
            if len(period_metrics) > 0:
                print(f"  • Средний размер группы: {period_metrics['active_instances'].mean():.1f}")
                print(f"  • Средняя утилизация: {period_metrics['avg_utilization'].mean()*100:.1f}%")
    
    # Сохранение результатов
    print(f"\n" + "=" * 70)
    request_df.to_csv('scenario2_requests.csv', index=False)
    metrics_df.to_csv('scenario2_metrics.csv', index=False)
    
    # Сохранение решений о масштабировании
    decisions_df = pd.DataFrame([
        {
            'time': d.timestamp,
            'action': d.action,
            'current_size': d.current_size,
            'target_size': d.target_size,
            'avg_utilization': d.avg_utilization
        }
        for d in autoscaler.scaling_decisions
    ])
    decisions_df.to_csv('scenario2_decisions.csv', index=False)
    
    print("Результаты сохранены в файлы:")
    print("• scenario2_requests.csv - детальная информация о запросах")
    print("• scenario2_metrics.csv - системные метрики")
    print("• scenario2_decisions.csv - решения о масштабировании")
    
    print(f"\nМоделирование завершено успешно!")

