from HawkesRLTradingV2.Orderbook import Order
class LimitLevel:
    def __init__(self, price: int):
        self.limitprice=price
        self.total_volume=0
        self.size=0
        self.head_order=None
        self.tail_order=None

    def add_order(self, order: Order):
        if not self.head_order or not self.tail_order:
            assert self.head_order is None
            assert self.tail_order is None
            self.head_order=self.tail_order=order
        else:
            self.tail_order.next=order
            order.prev=self.tail_order
            self.tail_order=order
        order.parentlimit=self
        self.size+=1
        self.total_volume+=order.size

    
    def remove_order(self, order: Order):
        if order.prev:
            order.prev.next=order.next
        else:
            self.head_order=order.next
        if order.next:
            order.next.prev=order.prev
        else:
            self.tail_order=order.prev
        self.size-=1
        self.total_volume-=order.size
    
