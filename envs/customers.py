import numpy as np
import random

class Customer:
    """顧客クラス"""
    def __init__(self, customer_id, position, seat_position):
        self.id = customer_id
        self.position = position
        self.seat_position = seat_position
        self.state = 'entering'
        self.wait_time = 0
        self.order_wait_time = np.random.randint(10, 30) # 注文までの時間
        self.has_ordered = False

class CustomerManager:
    """顧客生成・管理（最適化版）"""
    def __init__(self, enable_customers=True, spawn_interval=20, patience_limit=200):
        self.enable_customers = enable_customers
        self.spawn_interval = spawn_interval
        self.patience_limit = patience_limit
        self.customers = []
        self.customer_counter = 0
        self.steps_since_last_spawn = 0
    
    def spawn_customer(self, entrance_pos, seats):
        if not self.enable_customers or not seats or not entrance_pos:
            return
        occupied_seats = [c.seat_position for c in self.customers if c.state != 'leaving']
        available_seats = [s for s in seats if s not in occupied_seats]
        
        if available_seats:
            seat = random.choice(available_seats)
            customer = Customer(self.customer_counter, entrance_pos, seat)
            self.customers.append(customer)
            self.customer_counter += 1
    
    def update_customers(self):
        """戻り値: (新規注文リスト, キッチン投入リスト, キャンセル数)"""
        if not self.enable_customers: return [], [], 0
        
        active_orders, kitchen_items, cancel_count = [], [], 0
        
        for customer in self.customers[:]:
            customer.wait_time += 1
            
            # 忍耐限界によるキャンセル判定
            if customer.state != 'served' and customer.wait_time > self.patience_limit:
                self.customers.remove(customer)
                cancel_count += 1
                continue

            if customer.state == 'entering':
                customer.position = customer.seat_position
                customer.state = 'seated'
            elif customer.state == 'seated':
                if customer.wait_time >= customer.order_wait_time and not customer.has_ordered:
                    customer.state = 'ordered'
                    customer.has_ordered = True
                    active_orders.append(customer.seat_position)
                    kitchen_items.append({'time_left': 10})
            elif customer.state == 'served':
                if customer.wait_time > 30: # 食べ終わって退店
                    customer.state = 'leaving'
            elif customer.state == 'leaving':
                self.customers.remove(customer)
        
        return active_orders, kitchen_items, cancel_count