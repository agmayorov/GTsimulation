classdef TNode < handle
	properties
		data = []
	end
	properties (SetAccess = private, GetAccess = public)
		parent			TNode {mustBeScalarOrEmpty} = TNode.empty() 
		children (1,:)	TNode = TNode.empty() 
	end
	properties (Access = private)
		ref_list (1,1)	TList
	end
	
	methods
		function obj = TNode(data)
			arguments
				data = []
			end
			obj.data = data;
			obj.ref_list = TList();
			obj.ref_list.add_nodes(obj);
		end
		
	% Child Attaching and Creation
		
		function [obj, new_child] = attach_child(obj, new_child)
			% Attaches existing TNode object to this node as a child.
			%	Returns this node and the attached child node.
			arguments
				obj	TNode
				new_child (1,1) TNode
			end
			
			assert(~obj.is_same_tree_as(new_child), ...
				'Tree already contains specified TNode instance.');
			
			assert(~new_child.has_parent(), ...
				'Specified TNode is not a root node. Detach it from parent first.');
			
			new_child.parent = obj;
			obj.children(end + 1) = new_child;
			
			obj.ref_list.add_nodes(new_child.ref_list.list); 
			for node = new_child.ref_list.list
				node.ref_list = obj.ref_list;
			end
		end
		
		function [obj, new_child] = make_child(obj, data)
			% Creates a new child TNode object with the specified data.
			%	Returns this node and the new child node.
			arguments
				obj TNode
				data = []
			end
			new_child = TNode(data);
			obj.attach_child(new_child);
		end
		
	% Child Detaching
		
		function [obj, child] = detach_child(obj, child)
			% Detaches specified child node from this node.
			%	Creates a new tree with the detached node as its root.
			%	Returns this node and the detached node.
			arguments
				obj TNode
				child TNode
			end
			
			assert(child.parent == obj,	'TNode contains no specified child.');
			
			obj.children(obj.children == child) = [];
			child.parent = TNode.empty();
			
			detached_nodes = child.list_from_this();
			obj.ref_list.remove_nodes(detached_nodes); 
			
			new_list = TList(detached_nodes);
			for node = detached_nodes
				node.ref_list = new_list;
			end
			
		end
		
		function [obj, parent] = detach_from_parent(obj)
			% Detaches this node from its parent.
			%	Creates a new tree with the detached node as its root.
			%	Returns detached node and its ex-parent node.
			assert(obj.has_parent(), ... 
				'This TNode instance has no parent to detach from.');
			parent = obj.parent.detach_child(obj);			
		end

	% Node Relationships
		
		function root_node = root(obj)
			% Returns the root node of the tree.
			arguments
				obj (1,1) TNode
			end
			
			this = obj;
			while this.has_parent
				this = this.parent;
			end
			root_node = this;
		end
		
		function descendant_node = descendant(obj, vararg_child_numbers)
			% Returns the descendant node at the specified child number sequence.
			%	Calls: 
			%		t.descendant(1, 2, 3, ...)
			%		t.descendant([1, 2, 3, ...])
			%		t.descendant(1, [2, 3], ...)
			%	produce the same result and are equivalent to:
			%		t.children(1).children(2).children(3)...
			arguments
				obj TNode
			end
			arguments (Repeating)
				vararg_child_numbers double {mustBePositive, mustBeInteger}
			end
			
			child_nums = cell2mat(vararg_child_numbers);
			
			this = obj;
			for child_id = 1:length(child_nums)
				child_num = child_nums(child_id);
				try
					this = this.children(child_num);
				catch ME
					msg = sprintf( ...
						'Node at [this%s%s] does not have a child with index %d.', ...
						repmat(' -> ', [1, this ~= obj]), ...
						strjoin(strsplit(num2str(child_nums(1:child_id - 1))), ' -> '), child_num);
					new_exception = MException('MATLAB:TNode:no_valid_descendant', msg);
					ME = addCause(ME, new_exception);
					rethrow(ME)
				end
			end
			descendant_node = this;
		end
		
		function ancestor_node = ancestor(obj, generation)
			% Returns the ancestor node specified number of generations away from this.
			%	Equivalent to repeatedly calling 
			%		t.parent.parent....
			%	'generation' times.
			arguments 
				obj TNode
				generation (1,1) double {mustBeInteger, mustBePositive}
			end
			
			this = obj;
			for i = 1:generation
				assert(this.has_parent, 'Ancestor at generation %d from this has no parent.', i - 1);
				this = this.parent;
			end
			ancestor_node = this;
		end
		
		function bool = is_descendant_of(obj, other)
			% Checks whether this node is a descendant of the specified node.
			bool = false;
			this = obj;
			while this.has_parent
				this = this.parent; % obj skips itself.
				if this == other
					bool = true;
					return
				end
			end
		end
		
		function bool = is_ancestor_of(obj, other)
			% Checks whether this node is an ancestor of the specified node.
			bool = is_descendant_of(other, obj);
		end
		
	% Node Inspections
		
		function bool = has_parent(obj)
			% Checks whether this node has a parent (i.e. is not root node)
			bool = ~cellfun(@isempty, {obj.parent});
		end
		
		function bool = has_children(obj)
			% Checks whether this node has children (i.e. is not leaf node)
			bool = ~cellfun(@isempty, {obj.children});
		end
		
		function n = depth(obj)
			% Returns the distance from this node to the root node. Zero-based.
			n = arrayfun(@depth_scalar, obj);
		end
		
		function bool = is_same_tree_as(obj, node)
			% Checks whether this node belongs to the same tree as the other node.
			bool = obj.ref_list == node.ref_list;
		end
		
	% Search and Traversal
		
		function nodes = find_if(obj, predicate)
			% Finds nodes of the tree that satisfy the predicate.
			%	Searches through the whole tree 
			%	and returns an array of nodes for which predicate(node) == true.
			nodes = obj.ref_list.list(arrayfun(predicate, obj.list()));
		end
		
		function nodes = find_if_from_this(obj, predicate)
			% Finds nodes of a subtree that satisfy the predicate.
			%	Searches through a subtree starting at the current node (including it) 
			%	and returns an array of nodes for which predicate(node) == true.
			arguments
				obj TNode
				predicate function_handle
			end
			
			if predicate(obj)
				nodes = obj;
			else
				nodes = TNode.empty();
			end
			
			if ~obj.has_children()
				return
			end
			
			for child = obj.children
				nodes = [nodes, child.find_if_from_this(predicate)];
			end
		end
		
	% Listing
		
		function tree_list = list(obj)
			% Lists all nodes in a tree.
			%	Returns an array of all nodes in current tree structure.
			%	The order of the elements in a list has no meaning.
			%	Time complexity: Constant.
			tree_list = obj.ref_list.list;
		end
		
		function nodes = list_from_this(obj)
			% Lists all nodes of a subtree starting from this node.
			%	Returns an array of all nodes in a subtree.
			%	The order of the elements in a list is depth-first.
			%	Time complexity: O(n), where n - number of elements in the subtree.
			%	
			%	Use find_if_from_this() to search for elements in a subtree directly.
			nodes = find_if_from_this(obj, @(~)( true ));
		end
		
	% Tree and Subtree Copying
		
		function root_of_copy = copy(obj)
			% Copies all the nodes in the tree starting from root.
			%	Performs a deep copy of all the nodes. Returns the root node of the copied tree.
			%	The order of the nodes in 'children' array is preserved.	
			root_of_copy = obj.root().copy_from_this();
		end
		
		function root_of_copy = copy_from_this(obj)
			% Copies all the nodes in the subtree starting from this node.
			%	Performs a deep copy of all the nodes. Returns the root node of the copied tree.
			%	The order of the nodes in 'children' array is preserved.	
			root_of_copy = obj.copy_transform_from_this(@(x)( x ));
		end
		
		function root_of_copy = copy_transform(obj, transform)
			% Copies all the nodes in the tree starting from root node while transforming its data.
			%	Performs a deep copy of all the nodes. Returns the root node of the copied tree.
			%	Transformation is applied to the data of each node one-by-one in the form:
			%		copy.data = transform(original.data)
			%	The order of the nodes in 'children' array is preserved.	
			%
			%	Less memory intensive than copying the entire tree and then transforming each element.
			root_of_copy = obj.root().copy_transform_from_this(transform);
		end
		
		function root_of_copy = copy_transform_from_this(obj, transform)
			% Copies all the nodes in the subtree starting from this node while transforming its data.
			%	Performs a deep copy of all the nodes. Returns the root node of the copied tree.
			%	Transformation is applied to the data of each node one-by-one in the form:
			%		copy.data = transform(original.data)
			%	The order of the nodes in 'children' array is preserved.	
			%
			%	Less memory intensive than copying the entire subtree and then transforming each element.
			arguments
				obj TNode
				transform function_handle
			end
				
			root_of_copy = TNode(transform(obj.data));
			
			for child = obj.children
				root_of_copy.attach_child(child.copy_transform_from_this(transform));
			end
		end
		
	% Tree and Subtree Display
		
		function display_tree(obj, data_transform)
			% Prints out the tree in a cascading format to the console.
			%	The format of the output is:
			%		[depth](child_number/total_children) <custom_string>
			%	for node, from which this method was called, and
			%		[depth](child_number/total_children) custom_string
			%	for all other nodes.
			%
			%	The custom string by default is just "TNode" 
			%	but can be a function of each node's data, as:
			%		custom_string = data_transform(node.data);
			%	if data_transform function is specified by the user.
			arguments
				obj TNode
				data_transform function_handle = @(~)( "TNode" )
			end
			
			obj.root().display_from_this_and_highlight_other(obj, data_transform);			
		end
		
		function display_from_this(obj, data_transform)
			% Prints out the subtree in a cascading format to the console starting from this node.
			%	The format of the output is:
			%		[depth](child number/total children) <custom_string>
			%	for this node (root of the subtree) and
			%		[depth](child number/total children) custom_string
			%	for all other nodes.
			%
			%	The custom_string by default is just "TNode" 
			%	but can be a function of each node's data, as:
			%		custom_string = data_transform(node.data);
			%	if data_transform function is specified by the user.
			arguments
				obj TNode
				data_transform function_handle = @(~)( "TNode" )
			end
			
			obj.display_from_this_and_highlight_other(obj, data_transform);	
		end
		
		function display_from_this_and_highlight_other(obj, highlight_node, data_transform)
			% Prints out the subtree in a cascading format to the console starting from this node
			% and highlights the specified node.
			%	The format of the output is:
			%		[depth](child number/total children) <custom_string>
			%	for highlighted node and
			%		[depth](child number/total children) custom_string
			%	for all other nodes.
			%
			%	The custom_string by default is just "TNode" 
			%	but can be a function of each node's data, as:
			%		custom_string = data_transform(node.data);
			%	if data_transform function is specified by the user.
			arguments
				obj TNode
				highlight_node TNode
				data_transform function_handle = @(~)( "TNode" )
			end
			
			template_highlight_node = "%s[%d](%d/%d) %s\n";
			template_this_node      = "%s[%d](%d/%d) <%s>\n";
			
			% Because list_from_this() does a depth-first search, 
			% the order of elements in the returned list 
			% defines the hierarchy by depth alone. 
			% Listing ndoes sequentially and offseting each node by depth 
			% is enough to display the tree in a cascading view.
			
			ordered_list = obj.list_from_this(); 
			
			for node = ordered_list
				
				% Set position in children array and length of children array
				if node.has_parent
					siblings = node.parent.children;
					num_siblings = length(siblings);
					pos_in_siblings = find(siblings == node);
				else
					num_siblings = 1;
					pos_in_siblings = 1;
				end
				
				if node == highlight_node
					template = template_this_node;
				else
					template = template_highlight_node;
				end
				
				
				depth = node.depth();
				
				offset = repmat('    ', [1, depth]);
				
				fprintf(template, ...
					offset, depth, pos_in_siblings, num_siblings, data_transform(node.data)); 
				
			end
		end
		
		
	end
	
	methods (Access = private)
		function n = depth_scalar(obj)
			n = 0;
			this = obj;
			while this.has_parent
				this = this.parent;
				n = n + 1;
			end
		end		
	end
	
end

