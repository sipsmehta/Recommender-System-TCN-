# src/trust_network.py
import networkx as nx

class TrustNetwork:
    def __init__(self, trust_df):
        self.trust_df = trust_df
        self.trust_graph = None
        
    def build_trust_graph(self):
        """Build trust network graph"""
        self.trust_graph = nx.DiGraph()
        self.trust_graph.add_edges_from(
            self.trust_df[['trustor', 'trustee']].values
        )
        return self.trust_graph
    
    def get_trust_features(self, user_id):
        """Get trust network features for a user"""
        if user_id in self.trust_graph:
            return [nx.clustering(self.trust_graph, user_id)]
        return [0]
    
    def get_network_stats(self):
        """Get trust network statistics"""
        return {
            'nodes': self.trust_graph.number_of_nodes(),
            'edges': self.trust_graph.number_of_edges(),
            'avg_clustering': nx.average_clustering(self.trust_graph)
        }