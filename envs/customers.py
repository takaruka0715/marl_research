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
        self.order_wait_time = np.random.randint(5, 15)
        self.has_ordered = False
        self.served = False

class CustomerManager:
    """顧客生成・管理"""
    def __init__(self, enable_customers=True, spawn_interval=20):
        self.enable_customers = enable_customers
        self.spawn_interval = spawn_interval
        self.customers = []
        self.customer_counter = 0
        self.steps_since_last_spawn = 0
    
    def spawn_customer(self, entrance_pos, seats):
        if not self.enable_customers or len(seats) == 0:
            return
        occupied_seats = [c.seat_position for c in self.customers 
                         if c.state in ['seated', 'ordered', 'waiting_for_food']]
        available_seats = [s for s in seats if s not in occupied_seats]
        
        if len(available_seats) > 0 and entrance_pos:
            seat = random.choice(available_seats)
            customer = Customer(self.customer_counter, entrance_pos, seat)
            self.customers.append(customer)
            self.customer_counter += 1
    
    def update_customers(self):
        """顧客状態更新と注文管理"""
        if not self.enable_customers:
            return [], []
        
        active_orders = []
        kitchen_items = []
        
        for customer in self.customers[:]:
            if customer.state == 'entering':
                customer.position = customer.seat_position
                customer.state = 'seated'
            elif customer.state == 'seated':
                customer.wait_time += 1
                if customer.wait_time >= customer.order_wait_time and not customer.has_ordered:
                    customer.state = 'ordered'
                    customer.has_ordered = True
                    if customer.seat_position not in active_orders:
                        active_orders.append(customer.seat_position)
                        kitchen_items.append({'time_left': 5})
            elif customer.state == 'served':
                customer.wait_time += 1
                if customer.wait_time >= 15:
                    customer.state = 'leaving'
            elif customer.state == 'leaving':
                self.customers.remove(customer)
        
        return active_orders, kitchen_items
