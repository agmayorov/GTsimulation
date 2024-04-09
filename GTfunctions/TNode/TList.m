classdef TList < handle
	%TLIST Shared array of tree node references.
	properties (GetAccess = public, SetAccess = private)
		list (1,:) TNode = TNode.empty()
	end
	
	methods 
		function obj = TList(nodes)
			arguments
				nodes (1,:) TNode = TNode.empty()
			end
			obj.add_nodes(nodes);
		end
		
		function add_nodes(obj, nodes)
			arguments
				obj TList
				nodes (1,:) TNode
			end
			assert(~any(ismember(obj.list, nodes)), ...
				'TList alredy contains some of the specified nodes.');
			obj.list = [obj.list, nodes];
		end
		
		function remove_nodes(obj, nodes)
			arguments
				obj TList
				nodes (1,:) TNode
			end
			
			indecies = ismember(obj.list, nodes);
			assert(any(indecies), 'TList contains no specified nodes.');
			
			assert(sum(indecies) == length(nodes), ...
				'TList contains only some of the specified nodes.');
			
			obj.list(indecies) = [];
			
		end
		
	end
	
	
end

