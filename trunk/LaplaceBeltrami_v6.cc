/* ---------------------------------------------------------------------
 * $Id: step-3.cc 30147 2013-07-24 09:28:41Z maier $
 *
 * Copyright (C) 1999 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 *
 *          Info by Afonso:
 *
 *          This program is intended to evaluate the L2 and H1 errors from the solution by
 *          Nitsche's method in a circular boundary on a uniform background mesh (solved
 *          on step-3_vAfonso_2).
 *
 *          Att.: Vector<double> levelset returns positive values for DOF's
 *          at locations outside the boundary (defined by a radius)
 *          Limitations: initialization of point center and double radius are
 *          inside Function SignedDistanceCircle (want to declare these outside);
 *          maybe create a void set_center_radius ?
 *
 */



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <math.h> // log()
#include <utility> // pair
#include <vector> // std::vector

#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/base/derivative_form.h>
// If one wants to use ILU preconditioner
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/block_matrix_base.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/dofs/dof_renumbering.h>

#include "/home/afonsoal/Documents/dealii_8_2_1_unzipped/dealii-8.2.1/my_programs/my_includes/cut_cell_integration.h"
#include "/home/afonsoal/Documents/dealii_8_2_1_unzipped/dealii-8.2.1/my_programs/my_includes/NewCell.h"
#include "/home/afonsoal/Documents/dealii_8_2_1_unzipped/dealii-8.2.1/my_programs/my_includes/NewMesh.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_nothing.h>

#include <iomanip>
#include <cmath>
#include <limits>

#include <deal.II/grid/intergrid_map.h>
#include <set> // std::set

namespace cut_cell_method
{
using namespace dealii;

template <int dim>
class ExactSolution : public Function<dim>
{
public:

	ExactSolution () : Function<dim>(2) {} // In FAQ:  ExactSolution () : Function<dim>(dim+1) {}
	virtual ~ExactSolution(){}
	virtual void vector_value (const Point<dim>   &p,
			Vector<double>   &values ) const;

	virtual void vector_gradient (const Point<dim>   &p,
			std::vector<Tensor<1,dim>>  &gradient_values ) const;
};

template <int dim>
void ExactSolution <dim>:: vector_value (const Point<dim>   &p,
		Vector<double>   &values ) const	{

	// if fs = 108*..., do not divide by 12
	values(0) = 12*sin(3*atan2(p(1),p(0))) / 12;
//	values(0) = 12*(3*p(0)*p(0)*p(1)-p(1)*p(1)*p(1)) / 12;
	values(1) = (1*1-p.square())/4 ;

}

template <int dim>
void ExactSolution<dim>::vector_gradient (const Point<dim>   &p,
		std::vector<Tensor<1,dim>>  &gradient_values ) const	{
	Tensor<1,dim> return_value;

	// if fs = 108*..., do not divide by 12
	gradient_values[0][0] = (72*p(0)*p(1)) / 12;
	gradient_values[0][1] = (36*(p(0)*p(0)-p(1)*p(1))) / 12;

	gradient_values[1][0] = (-p(0)/2);
	gradient_values[1][1] = (-p(1)/2);
}
template <int dim>
class ExactSolutionUsurface : public Function<dim>
{
public:
	ExactSolutionUsurface () : Function<dim>() {} // In FAQ:  ExactSolution () : Function<dim>(dim+1) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

	virtual Tensor<1,dim> gradient (const Point<dim>   &p,
			const unsigned int  component = 0) const;
};

template <int dim>
double ExactSolutionUsurface <dim>:: value (const Point<dim>   &p,
		const unsigned int) const	{
	double return_value = 0;

//	return_value = 12*sin(3*atan2(p(1),p(0))); // ORIGINAL.
	return_value = 1*sin(3*atan2(p(1),p(0)));
	return return_value;
}

template <int dim>
Tensor<1,dim > ExactSolutionUsurface<dim>::gradient (const Point<dim>   &p,
		const unsigned int) const	{
	Tensor<1,dim> return_value;
	return_value[0] = 72*p(0)*p(1)/12;
	return_value[1] = 36*(p(0)*p(0)-p(1)*p(1))/12;
	return return_value; // return value = [ , ] (size = ?,dim)
}
template <int dim>
class ExactSolutionUbulk : public Function<dim>
{
public:
	ExactSolutionUbulk () : Function<dim>() {} // In FAQ:  ExactSolution () : Function<dim>(dim+1) {}

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;

	virtual Tensor<1,dim> gradient (const Point<dim>   &p,
			const unsigned int  component = 0) const;
};

template <int dim>
double ExactSolutionUbulk <dim>:: value (const Point<dim>   &p,
		const unsigned int) const	{
	double return_value = 0;

	return_value = (1*1-p.square())/4 ;
	return return_value;
}

template <int dim>
Tensor<1,dim > ExactSolutionUbulk<dim>::gradient (const Point<dim>   &p, const unsigned int) const {
	Tensor<1,dim> return_value;
	return_value[0] = (-p(0)/2);
	return_value[1] = (-p(1)/2);
	return return_value;
}
// Creation of Function to "weight" the non-solution elements (elements outside the circle)
template <int dim>
class WeightSolutionUbulk : public Function<dim>
{
public:
	WeightSolutionUbulk () : Function<dim>(2) {} // In FAQ:  ExactSolution () : Function<dim>(dim+1) {}

	virtual void vector_value (const Point<dim>   &p,
			Vector<double>   &values ) const;
};

template <int dim>
void WeightSolutionUbulk<dim>:: vector_value (const Point<dim>   &p,
		Vector<double>   &values ) const	{

	double return_value = 0;
	if (p.square() <= 1)
		return_value = 1;
	else
		return_value = 0;

	values[0] = 0;
	values[1] = return_value;
}

template <int dim>
class WeightSolutionUsurface : public Function<dim>
{
public:
	WeightSolutionUsurface () : Function<dim>(2) {} // In FAQ:  ExactSolution () : Function<dim>(dim+1) {}

	virtual void vector_value (const Point<dim>   &p,
			Vector<double>   &values ) const;
};

template <int dim>
void WeightSolutionUsurface<dim>:: vector_value (const Point<dim>   &p,
		Vector<double>   &values ) const	{

	// Function used to weight the error calculation. values[0] refers to the usurface var.,
	// while values[1] refers to the ubulk var. Note that this function doesn't care about domains,
	// only variables.

	// If this is uncommented, it means that all errors on nodes outside the real boundary (r=1)
	// are going to zero.
//	double return_value = (p.square()<=1) ? 1 : 0;
	// Best error Results are with weighting.
	double return_value = 1;
	values[0] = return_value;
	values[1] = 0;
}


template<int dim>
class SignedDistanceCircle_scalar: public Function<dim> {
	const double radius = 1.0;
	const Point<dim, double> center;
public:
	SignedDistanceCircle_scalar() :
		Function<dim>() {
	}
	virtual ~SignedDistanceCircle_scalar(){}
	virtual double value(const Point<dim> &p,
			const unsigned int component = 0) const;
};
//Represents a signed distance function from the circle.
template<int dim>
double SignedDistanceCircle_scalar<dim>::value(const Point<dim> &p,
		const unsigned int component) const {
	Assert(component == 0, ExcInternalError());
	//Center and radius of circle.
	double distance;
	// CIRCLE FUNCTION
	distance = p.distance(center) - radius;
	return distance;
}

template<int dim>
class SignedDistanceCircle: public Function<dim> {
	const double radius = 1.0;
	const Point<dim, double> center;
public:
SignedDistanceCircle () : Function<dim>(/*dim+1*/2) {}
virtual ~SignedDistanceCircle(){}
  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;
};

//Represents a signed distance function from the circle.
template<int dim>
void SignedDistanceCircle<dim>::vector_value (const Point<dim> &p,
	                                  Vector<double>   &values) const{
	//Center and radius of circle.
	double distance;
	distance = p.distance(center) - radius;
	// Levelset function returns the same, independent of the fe (ubulk or usurface)
	values(0) = distance;
	values(1) = distance;
}



template <int dim>
class PoissonProblem
{
public:
	PoissonProblem (int _n_cycles);
	void run ();

private:
	enum
	{
		not_surface_domain_id,
		surface_domain_id
	};

	enum
	{
		surface_block = 0,
		inside_block = 1
	};


	int ExtendedGlobal2Local (int,const typename hp::DoFHandler<dim>::cell_iterator &cell);
	static bool	cell_is_in_bulk_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
	static bool	cell_is_in_surface_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);
	void create_levelset_vector_map (std::map< double, Point<2> > & _levelset_map ,
			const Point<2> point);

	void make_grid ();
	void initialize_levelset();
	void output_results_levelset() const;
	void get_new_triangulation ();
	void set_active_fe_indices ();
	void setup_system_new ();
	void initialize_levelset_new();
	void assemble_system_newMesh ();
	void solve ();
	void output_results () const;
	void process_solution_ubulk();
	void interpolate_solution_usurface();

	int 				 cycle;
	int 				 n_cycles;
	Triangulation<2>     triangulation;
	Triangulation<2>  	 triangulation_new;
	FESystem<dim>        fe_surface;
	FESystem<dim>        fe_inside;
	FE_Q<dim>            fe_q_inside;

	FE_Q<2>              fe_justForMesh;
	FE_Q<1,2>				fe_dummy;
	hp::FECollection<dim> fe_collection_surface;
	DoFHandler<2>        dof_handler;
	hp::DoFHandler<2>        dof_handler_new;

    BlockSparsityPattern      	sparsity_pattern;
    BlockSparseMatrix<double> 	system_matrix;

    BlockSparsityPattern      	sparsity_pattern_new; // For some reason, this is based on _new here...
    BlockSparseMatrix<double> 	system_matrix_new;// For some reason, this is based on _new here...

    SparsityPattern 			sparsity_pattern_paraguaio;
    SparseMatrix<double>   		system_matrix_paraguaio;

    FullMatrix<double> 			FM_system_matrix;

    FullMatrix<double> 			FM_system_matrix_block00;
    FullMatrix<double> 			FM_system_matrix_block11;
    FullMatrix<double> 			FM_system_matrix_block11_yes_j;
    FullMatrix<double>			FM_system_matrix_block00_w_j;
    FullMatrix<double>			FM_system_matrix_block_yes_j_no_constraint;

    FullMatrix<double> FM_system_matrix_block00_yes_j_yes_constraint;
    FullMatrix<double> FM_system_matrix_block00_no_j_yes_constraint;


    // Complete FM_system_matrix with coupling, J_u and J_p added and constraint; RHS with constraint as well;
    FullMatrix<double> 			FM_system_matrix_block_w_constraint;
    Vector<double> 			    system_rhs_block01_w_constraint;
    Vector<double> 				solution_new_w_constraint;

    BlockVector<double>       solution;
    BlockVector<double>       solution_new; // For some reason, all is based on solution_new here...
    Vector<double>       solution_block0;
    Vector<double>       	solution_block1;
    Vector<double>       solution_block0_w_constraint;

    Vector<double> system_rhs_block0_w_constraint;

    BlockVector<double>       system_rhs;
    BlockVector<double>       system_rhs_new; // For some reason, all is based on solution_new here...
    Vector<double> system_rhs_paraguaio;


    Vector<double>       system_rhs_block0;
    Vector<double>       system_rhs_block1;

    ConstraintMatrix     constraints;
    MappingQ1 <2>	 	 mapping;


	FullMatrix<double>   	ALLERRORS_ubulk;
	FullMatrix<double>   	ALLERRORS_usurface;
	Vector<double>       	exact_solution_usurface;
	Vector<double>       	exact_solution_ubulk;

	Vector<double>       	difference_solution_usurface;
	Vector<double>       	difference_solution_ubulk;

	BlockVector<double> 	exact_solution;
	BlockVector<double>  	difference_solution;

	Vector<double>       levelset;

	SparseMatrix<double> mass_matrix;

	std::vector <bool>   isboundary_face;
	NewMesh 			 Obj_NewMesh;
	int n_dofs_surface, n_dofs_inside;
	hp::MappingCollection<dim> mapping_collection_surface;
	int new_n_cycles;
	std::map < double, Point<2> > levelset_face_map;
};

template <int dim>
PoissonProblem<dim>::PoissonProblem (int _n_cycles) // (const FiniteElement<dim> &fe)
:

fe_surface (FE_Q<dim>(1)      /*u_surface, @ surface*/, 1, FE_Q<dim>(1),/*u_bulk, @ surface*/ 1),
fe_inside  (FE_Nothing<dim>() /*u_surface, @ inside */, 1, FE_Q<dim>(1),/*u_bulk, @ inside */ 1),

fe_q_inside (1),

    fe_justForMesh (1),
    fe_dummy (1),
dof_handler ()
, dof_handler_new (triangulation_new),
new_n_cycles(_n_cycles)

//dof_handler (triangulation) // use original

{
	fe_collection_surface.push_back (fe_surface); // fe_index = 0
	fe_collection_surface.push_back (fe_inside);  // fe_index = 1
	mapping_collection_surface.push_back(mapping);
	mapping_collection_surface.push_back(mapping);
}

template <int dim>
int PoissonProblem<dim>::ExtendedGlobal2Local(int dof,
		const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
	int dof_index;
	int active_fe_index = cell->active_fe_index();
	if (dof<=7)
		dof_index = dof_handler_new.get_fe()[active_fe_index].system_to_component_index(dof).second;
	else if (dof == 8) 	dof_index = 4;
	else if (dof == 9)	dof_index = 4;
	else if (dof == 10)	dof_index = 5;
	else if (dof == 11)	dof_index = 5;
	else assert(0);

	assert(active_fe_index == 0);
	return dof_index;
}

template <int dim>
void PoissonProblem<dim>::create_levelset_vector_map (std::map< double, Point<2> > & _levelset_map ,
		const Point<2> point) {
	// Key represents the angle of point (x,y) with respect to the origin
	double key = atan2(point[1],point[0]);
	// This map will have the points of the level set function ordered by key, ie, from the smallest
	// to the biggest angle, such that the points joined will form an ordered circle. The order
	// is necessary to create the future mesh of the levelset triangulaiton (used for solution
	// interpolation and error evaluation)
	if (_levelset_map.count(key) == 0)
		_levelset_map[key] = point;
}


template <int dim>
bool PoissonProblem<dim>:: cell_is_in_bulk_domain
(const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
  return (cell->material_id() == not_surface_domain_id);
}


template <int dim>
bool PoissonProblem<dim>::cell_is_in_surface_domain
(const typename hp::DoFHandler<dim>::cell_iterator &cell)
{
  return (cell->material_id() == surface_domain_id);
}

template <int dim>
void PoissonProblem<dim>::make_grid ()
{
	if (cycle == 0) // This assures that the Grid is created only once, i.e., in the first cycle;
	{
		GridGenerator::hyper_cube (triangulation, -2, 2);
		triangulation.refine_global(3);// Now, it just makes sense to start in 2
	}

//	int refinement_global = 1;
	int refinement_global = new_n_cycles;

	triangulation.refine_global(refinement_global); // Last refinement cycle
	dof_handler.initialize (triangulation,/*fe*/fe_justForMesh);
	constraints.close();


	dof_handler.distribute_dofs (/*fe*/ fe_justForMesh);


	std::string filename_new = "triangulation-";
	filename_new += ('0' + cycle);
	filename_new += ".eps";

	std::ofstream out (filename_new.c_str());
	GridOut grid_out;
	grid_out.write_eps (triangulation, out);
	std::cout << "First Triangulation created \n";
	std::cout << "Refinement #: "<< refinement_global+3 << "\n";



}

template <int dim>
void PoissonProblem<dim>::initialize_levelset() {
	levelset .reinit(/*n_dofs_surface*/ dof_handler.n_dofs());

	VectorTools::interpolate(dof_handler, SignedDistanceCircle_scalar<dim>(),
			levelset);
}

template <int dim>
void PoissonProblem<dim>::output_results_levelset() const {
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(levelset, "levelset");
	data_out.build_patches();

	std::string filename = "levelset-";
	filename += ('0' + cycle);
	filename += ".vtk";
	std::ofstream output(filename.c_str());
	data_out.write_vtk(output);
}

template <int dim>
void PoissonProblem<dim>::get_new_triangulation ()
{
	QGauss<dim>  quadrature_formula(2);
	QGauss<1> face_quadrature_formula(2);

	FEValues<dim> fe_values (/*fe*/fe_justForMesh, quadrature_formula,
			update_values | update_gradients | update_JxW_values
			| update_quadrature_points | update_jacobians |
			update_support_jacobians | update_inverse_jacobians);

	FEFaceValues<dim> fe_face_values (/*fe*/fe_justForMesh, face_quadrature_formula,
			update_values |
			update_gradients |
			update_quadrature_points  |
			update_normal_vectors | update_JxW_values
			| update_hessians );


	const unsigned int   dofs_per_cell = /*fe*/ fe_justForMesh.dofs_per_cell;
//	std::cout << "dofs_per_cell: " << dofs_per_cell << "\n"; // Output: 8, ok
	int count_cell = 0;

	std::vector<types::global_dof_index> cell_global_dof_indices (dofs_per_cell);


	DoFHandler<2>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();


	// Creates a vector support_points [n.dofs() x 2] which has the x [0] and y [1]
	// coordinates for each (global) DOF (line)
	std::vector<Point<dim> > support_points(dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);



	bool isinside = false;
	bool isboundary = false;
	Obj_NewMesh.reinit();
	Obj_NewMesh.set_variables(support_points,dofs_per_cell);
	isboundary_face.clear();
	Point<2> VOID_POINT (-10000,-10000);
	std::vector<types::global_dof_index> face_dof_indices (2);
	for (; cell!=endc; ++cell)
	{
		fe_values.reinit (cell);
		cell->get_dof_indices(cell_global_dof_indices);

		isinside = false;
		isboundary = false;

		int inside = 0;
		int coincident_node = 0;

		// Identify inside cells (case A)

		for (unsigned int i=0;i<dofs_per_cell;++i) {
			unsigned int j = cell_global_dof_indices[i];

			if (levelset[j]<=0) {
				inside++;
				if (levelset[j] == 0) {
					coincident_node++;
				}
			}
		}
		// Case A: cell is entirely inside boundary
		if (inside == 4) {
			isinside = true;
			//						std::cout << "Inside cell: "<< isinside << "\n";
		}
		// Case B: cell is entirely outside boundary, even if 1 node is coincident w/ boundary
		if (inside == 0) {
//			isoutside = true;
			//						std::cout << "Outside cell op1: "<< isoutside << "\n";
		}
		if (inside == 1 && coincident_node == 1){
//			isoutside = true;
			//						std::cout << "Outside cell op2: "<< isoutside << "\n";
		}
		// Case C: cell has 3 nodes inside boundary, i.e., is a boundary cell.
		if (inside > 1 && inside < 4 ) {
			isboundary = true;
			//						std::cout << "Boundary cell op1: "<< isboundary << "\n";
			//			integrate_new_cut_face = true;
		}
		if (inside == 1 && coincident_node != 1 ) {
			isboundary = true;
			//						std::cout << "Boundary cell op2: "<< isboundary << "\n";
			//			integrate_new_cut_face = true;
		}
		/*		if (inside == 2 && coincident_node == 2) {
			isboundary = false;
			isinside = false;
//		    std::cout << "Outside special \n";
		}
		if (inside == 4 && coincident_node >= 2) {
			isboundary = true;
			isinside = false;
//		    std::cout << "Isboundary special \n";
		    integrate_new_cut_face = true;
		}
		if (inside == 4 && coincident_node > 2)
			integrate_new_cut_face = false;*/

		if (isboundary) isboundary_face.push_back(true);
		else isboundary_face.push_back(false);

		bool cell_is_boundary;
		if(isinside)	{
			cell_is_boundary = false;
			Obj_NewMesh.set_new_mesh(cell_global_dof_indices,
					cell_is_boundary);
		}
		if(isboundary)	{
			cell_is_boundary = true;
			Obj_NewMesh.set_new_mesh(cell_global_dof_indices,
					cell_is_boundary);
		}
		count_cell++;
	} // (end for cell)

	triangulation_new.clear();
	Obj_NewMesh.create_new_triangulation(triangulation_new);
	std::string filename_new = "triangulation_new-";
	filename_new += ('0' + cycle);
	filename_new += ".eps";
	std::ofstream out (filename_new);
	GridOut grid_out;
	grid_out.write_eps (triangulation_new, out);
	std::cout << "New Triangulation created: triangulation_new \n";
}

template <int dim>
void PoissonProblem<dim>::set_active_fe_indices ()
{
	int count_cell = 0;
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation_new.begin_active();
         cell != triangulation_new.end(); ++cell) {
      if (Obj_NewMesh.vector_new_cell_struct[count_cell].cell_is_boundary) {
        cell->set_material_id (surface_domain_id);
      }
      else
        cell->set_material_id (not_surface_domain_id);
      count_cell++;
    }

    int inside_cells = 0, surface_cells = 0;
  for (typename hp::DoFHandler<dim>::active_cell_iterator
       cell = dof_handler_new.begin_active();
       cell != dof_handler_new.end(); ++cell)
    {
      if (cell_is_in_bulk_domain(cell)){
        cell->set_active_fe_index (1);
        inside_cells++;
      }
      else if (cell_is_in_surface_domain(cell)) {
        cell->set_active_fe_index (0);
        surface_cells++;
      }
      else
        Assert (false, ExcNotImplemented());
    }
  	std::cout<<" number of cells (inside/surface): "<< inside_cells << " ; "<<surface_cells <<std::endl;
}

template <int dim>
void PoissonProblem<dim>::setup_system_new (){
	set_active_fe_indices();
	dof_handler_new.distribute_dofs (fe_collection_surface);
    DoFRenumbering::Cuthill_McKee (dof_handler_new);

    std::vector<unsigned int> block_component (2,0);
    block_component[0] = surface_block;
    block_component[1] = inside_block;

    DoFRenumbering::block_wise(dof_handler_new);

    std::vector<types::global_dof_index> dofs_per_block (2);
    DoFTools::count_dofs_per_block (dof_handler_new, dofs_per_block, block_component);
    n_dofs_surface = dofs_per_block[0];
    n_dofs_inside = dofs_per_block[1];

//     fe_collection_surface(0) -> 	fe_surface
//     								fe_surface(0) -> FE_Q 		(Var. usurface @ surface)
//     								fe_surface(1) -> FE_Q 		(Var. ubulk @ surface)
//     fe_collection_surface(1) -> 	fe_inside
//     								fe_inside(0) -> FE_NOTHING 	(Var. usurface @ surface = {0})
//     								fe_inside(1) -> FE_Q		(Var. ubulk @ inside)

    /* fe_collection_surface(0)(0) (USURFACE) and fe_collection_surface(0)(1) (UBULK) are coupled,
     * so one can couple the equations normally. Just follow methodology by step-20_doubleLaplace 3
     * or step-46.
     * For the equations INSIDE the domain, one must basically solve the equations of
     * cell->active_fe_index() == 1 (fe_collection_surface(1)(1) with the usual FE formulation
     * (no cut cell etc. needed)
     * # DOF's breakdown for a 4 cycle refinement:
     * 	(n_dofs_surface: 56 + n_dofs_inside: 77);
     * 	fe_collection_surface(1)    dofs: 45
     * 	fe_collection_surface(0)(0) dofs: 56
     * 	fe_collection_surface(0)(1) dofs: 56
     * 	n_dofs_inside: 77 refers to inside (fe_collection_surface(1)) dofs 45 +
     * 	surface dofs (fe_collection_surface(0)(1)) 56, minus the coincident nodes = 77.
     * 	This is the block 1,1. The block 0,0 is the surface, fe_collection_surface(0)(0) only (USURFACE).
     * 	The coupling occurs naturally. If one wants to solve the variables uncoupled,
     * 	 one must solve each block alone.
    * */
    /*The DOF numbering occurs like this:
     * (unit cell dof's)	(global cell dof's)
     * K cell @ surface
     * 4,5  6,7					* 12,52  13,54
     *   ___ 					*  	   ___
     *  |   | 					*  	  |   |
     *  |___|					*  	  |___|
     * 0,1  2,3 				* 10,50  11,51
     *
     * 0,2,4,6 refer to the DOF's beloging to fe_collection_surface(0)(0) (usurface)
     * 1,3,5,7 refer to the DOF's beloging to fe_collection_surface(0)(1) (ubulk)
     *
     * (unit cell dof's)		(global cell dof's)
     * K'  cell @ surface
     * 2    3					* 	54     56
     *   ___ 					*  	   ___
     *  |   | 					*  	  |   |
     *  |___|					*  	  |___|
     * 0     1 					*   51    55
     *
     * Usual 1 element dof distribution for fe_collection_surface(1)(1) (usurface)
     * recall that fe_collection_surface(1)(0) doesn't have any dof.
     * In this example, K and K' are neighbors whereas K lies on the surface domain and K' in the
     * inside domain. Note that the DOF numbering of K' follows the same as K, for the DOF's belonging
     * to fe_collection_surface(0)(1). Indeed, they form the block 1,1 as explained before.
     * This setup is good because enables easy coupling of fe_collection_surface(0)(0) and (0)(1)
     * and a block assembly of each variable; it makes it easy to solve the uncoupled problem, as well
     * as inserting Lagrange multipliers in the stiffness matrix.
     *
     * */

    constraints.clear ();

    std::cout << "Number of degrees of freedom: "
    		<< dof_handler_new.n_dofs()
    		<< " (n_dofs_surface: " << n_dofs_surface << " + n_dofs_inside: "<< n_dofs_inside  <<')'
    		<< std::endl;

    std::cout << "Number of active cells: "
              << triangulation_new.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation_new.n_cells()
              << std::endl;

    constraints.close();
    //step-22 like
//    BlockSparsityPattern      	sparsity_pattern;
    {
    	// From step 11:
    	/*It does not require that we know in advance how many entries rows could have,
    	 * but allows just about any length. It is thus significantly more flexible in
    	 * case you do not have good estimates of row lengths,*/

    	BlockCompressedSimpleSparsityPattern csp (2,2);
    	csp.block(0,0).reinit (n_dofs_surface, n_dofs_surface);
    	csp.block(1,0).reinit (n_dofs_inside, n_dofs_surface);
    	csp.block(0,1).reinit (n_dofs_surface, n_dofs_inside);
    	csp.block(1,1).reinit (n_dofs_inside, n_dofs_inside);

    	csp.collect_sizes();

    	// From step-22:
    	/*The way we are going to do that is to pass the information about constraints to the
    	 *  function that generates the sparsity pattern, and then set a false argument specifying
    	 *   that we do not intend to use constrained entries:
    	 *   DoFTools::make_sparsity_pattern (dof_handler, csp, constraints, false);
    	 *   This functions obviates, by the way, also the call to the condense() function
    	 *   on the sparsity pattern.*/
    	//	    DoFTools::make_sparsity_pattern (dof_handler_new, csp, constraints, false);
    	DoFTools::make_sparsity_pattern (dof_handler_new, csp/*, constraints, false*/);
    	// From step-11:
    	/*As a further sidenote, you will notice that we do not explicitly have to compress the
    	 * sparsity pattern here. This, of course, is due to the fact that the copy_from function
    	 *  generates a compressed object right from the start, to which you cannot add new entries
    	 *  anymore. The compress call is therefore implicit in the copy_from call.*/

    	sparsity_pattern.copy_from (csp);
    	// END step-22 like
    }
    {
    	system_matrix_new.reinit (sparsity_pattern);

    	CompressedSparsityPattern c_sparsity(dof_handler_new.n_dofs());
    	DoFTools::make_sparsity_pattern (dof_handler_new, c_sparsity);
    }

        solution_new.reinit (2);
	    solution_new.block(0).reinit (n_dofs_surface);
	    solution_new.block(1).reinit (n_dofs_inside);
	    solution_new.collect_sizes ();
	    exact_solution.reinit (2);
	    exact_solution.block(0).reinit (n_dofs_surface);
	    exact_solution.block(1).reinit (n_dofs_inside);
	    exact_solution.collect_sizes ();
	    difference_solution.reinit (2);
	    difference_solution.block(0).reinit (n_dofs_surface);
	    difference_solution.block(1).reinit (n_dofs_inside);
	    difference_solution.collect_sizes ();
	    exact_solution_usurface.reinit (n_dofs_surface);
	    exact_solution_ubulk.reinit (n_dofs_inside);

	    difference_solution_usurface.reinit (n_dofs_surface);
	    difference_solution_ubulk.reinit (n_dofs_inside);
	    system_rhs_new.reinit (2);
	    system_rhs_new.block(0).reinit (n_dofs_surface);
	    system_rhs_new.block(1).reinit (n_dofs_inside);
	    system_rhs_new.collect_sizes ();

	    // Create solution blocks and augmented global vector to enforce constraints.
		solution_block0.reinit(solution_new.block(0).size());
		solution_block1.reinit(solution_new.block(1).size());
		solution_block0_w_constraint.reinit(n_dofs_surface+1);
		solution_new_w_constraint.reinit(dof_handler_new.n_dofs()+1);
}

template <int dim>
void PoissonProblem<dim>::initialize_levelset_new() {
	// Need to reinitialize the levelset function with different size, corresponding
	// to the new triangulation
	levelset.reinit(dof_handler_new.n_dofs());
	VectorTools::interpolate(dof_handler_new, SignedDistanceCircle<dim>(),
			levelset);
}

template <int dim>
void PoissonProblem<dim>::assemble_system_newMesh ()
{
	QGauss<dim>  quadrature_formula(2);
	QGauss<1> face_quadrature_formula(2);

    hp::QCollection<dim>  q_collection;
    q_collection.push_back (quadrature_formula);
    q_collection.push_back (quadrature_formula);

    hp::FEValues<dim> 	 hp_fe_values (fe_collection_surface,
    								q_collection ,
    								update_values | update_gradients | update_JxW_values
    								| update_quadrature_points | update_jacobians | update_support_jacobians
    								| update_inverse_jacobians);

    FEFaceValues<dim>    fe_face_values( fe_surface ,
    											face_quadrature_formula,
                                                update_JxW_values |
                                                update_normal_vectors |
                                                update_gradients);

    hp::FEValues<dim>    fe_values_neighborCell (fe_collection_surface,
    												q_collection ,
    												update_values | update_gradients | update_JxW_values
    												| update_quadrature_points | update_jacobians | update_support_jacobians
    												| update_inverse_jacobians);

    // FE object used for the stabilization term, where the face of the neighbor cell is in a
    // surface cell
    FEFaceValues<dim>   fe_face_values_neighborCell_surface (fe_surface,
            											face_quadrature_formula,
                                                        update_JxW_values |
                                                        update_normal_vectors |
                                                        update_gradients);
    // FE object used for the stabilization term, where the face of the neighbor cell is in a
    // cell inside (not surface). The appropriate object will be chosen based on the location of the
    // neighbor cell.
    FEFaceValues<dim>    fe_face_values_neighborCell_inside (fe_inside,
                											face_quadrature_formula,
                                                            update_JxW_values |
                                                            update_normal_vectors |
                                                            update_gradients);

	const unsigned int   n_q_points    = quadrature_formula.size();
	int count_cell = 0;

	FullMatrix<double>   cell_matrix /*(dofs_per_cell, dofs_per_cell)*/;
	Vector<double>       cell_rhs /*(dofs_per_cell)*/;
	FullMatrix<double>   cell_j_matrix /*(dofs_per_cell+4 // 2 here, dofs_per_cell+4)*/;
	FullMatrix<double>   cell_j_matrix_p /*(dofs_per_cell+4, dofs_per_cell+4)*/;

	std::vector<types::global_dof_index> local_dof_indices /*(dofs_per_cell)*/;

	// the "traditional" cell_global_dof_indices" is now exclusive to the u variable,
	// and the vector of all variables (u, p) is _ALL.
	std::vector<types::global_dof_index> cell_global_dof_indices_ALL /*(dofs_per_cell)*/;
	std::vector<types::global_dof_index> cell_global_dof_indices     /*(dofs_per_cell/2)*/;
	std::vector<types::global_dof_index> cell_global_dof_indices_usurface 	 /*(dofs_per_cell/2)*/;
	std::vector<types::global_dof_index> cell_global_dof_indices_ubulk 	 /*(dofs_per_cell/2)*/;

	std::vector<types::global_dof_index> face_dof_indices_ALL /*(dofs_per_face)*/;
	std::vector<types::global_dof_index> face_dof_indices     /*(dofs_per_face/2)*/;
	std::vector<types::global_dof_index> face_dof_indices_usurface   /*(dofs_per_face/2)*/;
	std::vector<types::global_dof_index> face_dof_indices_ubulk   /*(dofs_per_face/2)*/;

	std::vector<types::global_dof_index> neighbor_cell_global_dof_indices_ALL /*(dofs_per_cell)*/;
	std::vector<types::global_dof_index> neighbor_cell_global_dof_indices /*(dofs_per_cell/2)*/;
	std::vector<types::global_dof_index> neighbor_cell_global_dof_indices_usurface /*(dofs_per_cell/2)*/;
	std::vector<types::global_dof_index> neighbor_cell_global_dof_indices_ubulk /*(dofs_per_cell/2)*/;

	std::vector<types::global_dof_index> neighbor_face_dof_indices_ALL /*(dofs_per_face)*/;
	std::vector<types::global_dof_index> neighbor_face_dof_indices /*(dofs_per_face/2)*/;
	std::vector<types::global_dof_index> neighbor_face_dof_indices_usurface /*(dofs_per_face/2)*/;
	std::vector<types::global_dof_index> neighbor_face_dof_indices_ubulk /*(dofs_per_face/2)*/;

	std::vector<Point<dim> > support_points(dof_handler_new.n_dofs());
	DoFTools::map_dofs_to_support_points(mapping_collection_surface, dof_handler_new, support_points);

	std::vector<double > quadrature_weights = face_quadrature_formula.get_weights();
	std::vector<Point<1> > face_quadrature_points = face_quadrature_formula.get_points();
	std::vector<Point<2> > cell_quadrature_points = quadrature_formula.get_points();
	std::vector<double > cell_quadrature_weights = quadrature_formula.get_weights();

	std::vector<int> j_integrate_face;
	std::vector<int> j_integrate_face_p;
	std::vector< unsigned int > FULL_EXTENDED_global_dof_indices;

	std::vector< Point<2> > new_face_vector;
	std::vector< Point<2> > j_face_vector;
	std::vector< Point<2> > j_face_vector_usurface;
	std::vector< Point<2> > all_points_vector;
	std::vector< unsigned int  > j_face_vector_global_dofs;
	std::vector< unsigned int  > j_face_vector_global_dofs_p;

	std::vector< Point<2> > new_normal_vector;

	FullMatrix<double> j_matrix (n_dofs_surface,n_dofs_surface);

	FullMatrix<double> j_matrix_p (n_dofs_inside,n_dofs_inside);

	Point<2> VOID_POINT (-10000,-10000);
	unsigned int VOID_INT = -10000;
	double cell_diameter;
	// Parameter alfa = gamma_D*h⁻1 (Refer to Burman (2012) and Mendez (2004))
	double gamma_D = 5;

	double fs;
	// From Burman et al 2014 (Cut FEM...)
	const double 	gamma_1 = 0.01; // 0.01; // 0.1; // multiplier of stab. term of usurface
	std::cout << "Gamma_1 = " << gamma_1 << "\n";
	const double gamma_p = /*.01;*/   0.1; // multiplier of stab. term of ubulk
	const double b_B = 1 ;
	const double b_S = 1 ;
	const double k_B = 1;
	const double k_S = 1;

	const double g_D = 0; // Value of Dirichlet boundary
	const double f_B = 1;
	std::vector<types::global_dof_index> ::const_iterator first;
	std::vector<types::global_dof_index> ::const_iterator last;
	int count_u = 0, count_p = 0;

	int dof_index_i, dof_index_j;

	const FEValuesExtractors::Scalar usurface (0);
	const FEValuesExtractors::Scalar ubulk 	  (1);

	// Create constraint vector
//	Vector<double>       constraint_vector (dof_handler_new.n_dofs());
	BlockVector<double>  constraint_vector (dof_handler_new.n_dofs());
	constraint_vector.reinit (2);
	constraint_vector.block(0).reinit (n_dofs_surface);
	constraint_vector.block(1).reinit (n_dofs_inside);
	constraint_vector.collect_sizes ();

	hp::DoFHandler<2>::active_cell_iterator
	cell = dof_handler_new.begin_active(),
	endc = dof_handler_new.end();
	WeightSolutionUbulk<dim> extract_weight_exact_solution;
	ExactSolutionUbulk<dim> extract_exact_solutionUbulk;
	ExactSolutionUsurface<dim> extract_exact_solutionUsurface;
	for (; cell!=endc; ++cell)
	{
		// Output to a all_points_vector all the points (nodes) of the new mesh, to vis. in Gnuplot

//		for (unsigned int face=0; face<GeometryInfo<2>::faces_per_cell; ++face)
//		{
//			fe_face_values.reinit (cell, face);
//			cell->face(face)->get_dof_indices(face_dof_indices_ALL);
//			count_u = 0;
//			for (unsigned int i=0; i<face_dof_indices_ALL.size(); ++i) {
//				if (dof_handler_new.get_fe()[active_fe_index].
//									system_to_component_index(i).first == 0) {
//					face_dof_indices[count_u] = face_dof_indices_ALL[i];
//					count_u++;
//				}
//				for (unsigned int i=0;i<2;++i) {
//					unsigned int j = face_dof_indices[i];
//					all_points_vector.push_back(support_points[j]);
//				}
//				all_points_vector.push_back(VOID_POINT);
//			}
//		}

		//		std::cout << "cell #: " << cell << "\n";
		hp_fe_values.reinit (cell);
		const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
		int dofs_per_cell = cell->get_fe(	).dofs_per_cell;
		local_dof_indices.resize(dofs_per_cell);

		cell_matrix.reinit (dofs_per_cell,dofs_per_cell);
		cell_matrix = 0;
		cell_rhs.reinit (dofs_per_cell);
		cell_rhs = 0;

		int active_fe_index = cell->active_fe_index();


		// Assemble matrix inside using the usual FE formulation.
		if (cell_is_in_bulk_domain(cell))	{
			assert(active_fe_index == 1);
			cell_global_dof_indices_ALL.resize(dofs_per_cell);
			cell->get_dof_indices(cell_global_dof_indices_ALL);


			for (unsigned int q_point=0; q_point<n_q_points; ++q_point) {
				for (unsigned int i=0; i<dofs_per_cell; ++i) {
					for (unsigned int j=0; j<dofs_per_cell; ++j) {
						cell_matrix(i,j) += (fe_values.shape_grad(i, q_point) *
								/*shape_grad*/
								fe_values.shape_grad /*shape_grad*/
								(j, q_point) *
								fe_values.JxW (q_point));
					}
					cell_rhs(i) += (fe_values.shape_value (i, q_point) *
							f_B * fe_values.JxW (q_point));
					unsigned int j = cell_global_dof_indices_ALL[i];
					exact_solution_ubulk[j-n_dofs_surface] =  extract_exact_solutionUbulk.value(support_points[j])
//						 *extract_weight_exact_solution.value(support_points[j])
						 ;
				}
			}
		}

		// Assemble terms on the surface domain.
		if (cell_is_in_surface_domain(cell)) {
			assert(active_fe_index == 0);

		int	dofs_per_face = cell->get_fe().dofs_per_face;

		face_dof_indices_ubulk.resize(dofs_per_face);
		neighbor_face_dof_indices.resize(dofs_per_face);

		cell_j_matrix.reinit (dofs_per_cell+/*2*/4,dofs_per_cell+/*2*/4);
		cell_j_matrix_p.reinit (dofs_per_cell+/*2*/4,dofs_per_cell+/*2*/4);

		cell_global_dof_indices_ALL.resize(dofs_per_cell);
		cell_global_dof_indices.resize(dofs_per_cell/2);
		cell_global_dof_indices_usurface.resize(dofs_per_cell/2);
		cell_global_dof_indices_ubulk.resize(dofs_per_cell/2);

		cell->get_dof_indices(cell_global_dof_indices_ALL);

		face_dof_indices_ALL.resize(dofs_per_face);
		face_dof_indices.resize(dofs_per_face/2);
		face_dof_indices_usurface.resize(dofs_per_face/2);
		face_dof_indices_ubulk.resize(dofs_per_face/2);

		neighbor_cell_global_dof_indices_ALL.resize(dofs_per_cell);
		neighbor_cell_global_dof_indices.resize(dofs_per_cell/2);
		neighbor_cell_global_dof_indices_usurface.resize(dofs_per_cell/2);
		neighbor_cell_global_dof_indices_ubulk.resize(dofs_per_cell/2);

		neighbor_face_dof_indices_ALL.resize(dofs_per_face);
		neighbor_face_dof_indices.resize(dofs_per_face/2);
		neighbor_face_dof_indices_usurface.resize(dofs_per_face/2);
		neighbor_face_dof_indices_ubulk.resize(dofs_per_face/2);


		// Separate cell_global_dof_indices into vector of global dof indices
		// exclusively for each one of the variables.
		// Example:
		// local dof_i,j			           :   0 ,1 ,2 ,3 ,4 ,5 ,6 ,7
		// cell_global_dof_indices_ALL (size=8):   0 ,77,1 ,78,2 ,79,3 ,80
		// cell_global_dof_indices (u) (size=4):   0 ,   1 ,   2    ,3
		// cell_global_dof_indices_ubulk   (size=4):      77,  ,78   ,79   ,80

		// Update: This could have been done much more elegantly following:
		// https://groups.google.com/forum/#!searchin/dealii/remove$20cells/dealii/0y93BHXL10M/840shoGzMP8J
		// If I understand you correctly, it is sufficient for you to get the support points of the dofs.
		// For this you can use a quadrature formula that is initialized with
		// fe.get_unit_support_points. Then use a FEValues object that updates the quadrature points
		// and call get_quadrature_points on each cell. The output is a std::vector and the entries
		// are the support points of the local dofs.
		count_u = 0;
		count_p = 0;
		for (unsigned int i=0; i<cell_global_dof_indices_ALL.size(); ++i) {
			if (dof_handler_new.get_fe()[active_fe_index].system_to_component_index(i).first == 0) {
				cell_global_dof_indices[count_u] = cell_global_dof_indices_ALL[i];
				count_u++;
			}
			if (dof_handler_new.get_fe()[active_fe_index].system_to_component_index(i).first == 1) {
				cell_global_dof_indices_ubulk[count_p] = cell_global_dof_indices_ALL[i];
				count_p++;
			}
		}

		// Input the cell_diameter for the error calculation, see process_results
		cell_diameter = cell->diameter();
		ALLERRORS_ubulk[cycle][3] = cell_diameter;
		ALLERRORS_usurface[cycle][3] = cell_diameter; // This is in class assemble_system


		double alfa = gamma_D/cell_diameter; // With this, the parameter alfa will change for each cycle!

		int integrate_face_check = 0;  // keep track of the number of the integrated
		// faces of the cell, i.e., new cut faces and other boundary faces of the cell
		int intersected_face = 0; // keep track of the number of intersected faces;
		int inside_faces = 0; // keep track of the number of inside faces;
		int calls_to_j = 0; // test number of calls to integrate stab. j term.
		// calls_to_j should be equal to intersected_face+inside_faces

		cut_cell_integration Obj_cut_cell_integration
		(fe_values,/*fe*/FE_Q<2>(1),quadrature_formula,face_quadrature_formula);

			NewCell ObjNewCell_new(levelset);

			std::vector<Point<2> > new_face(2);

			// Loop over faces, to identify faces
			for (unsigned int face=0; face<GeometryInfo<2>::faces_per_cell; ++face)
			{
				fe_face_values.reinit (cell, face);

				// We need to specify which DOF indices from this face we want: the face may have
				// different FE associated to it, so different DOF numbering.
				cell->face(face)->get_dof_indices(face_dof_indices_ALL,active_fe_index);
				count_u = 0;
				count_p = 0;
				for (unsigned int i=0; i<face_dof_indices_ALL.size(); ++i) {
					if (dof_handler_new.get_fe()[active_fe_index].system_to_component_index(i).first
							== 0) {
						face_dof_indices[count_u] = face_dof_indices_ALL[i];
						count_u++;
					}
					if (dof_handler_new.get_fe()[active_fe_index].system_to_component_index(i).first
							== 1) {
						face_dof_indices_ubulk[count_p] = face_dof_indices_ALL[i];
						count_p++;
					}
				}


				bool integrate_face_usurface = false;
				bool integrate_face_ubulk = false;

				unsigned int k0 = face_dof_indices[0] ; // Global DOF
				unsigned int k1 = face_dof_indices[1] ;


				//	INTEGRATE THE 2 FACES INTERSECTED by the boundary.
				//	Have to actually create 2 new faces.
				//  Cells with coincident nodes are considered in the next if{}
				if ( ((levelset[k0]*levelset[k1]) < 0.0) /*|| bool_test*/)
					// Face is intersected by the Boundary
				{
					Point<dim> X0;
					Point<dim> X1;
					Point<dim> face_normal_vector;
					X1 = ObjNewCell_new.getIntersection(support_points[k0],
							support_points[k1], k0,k1);

					// Defining new pair of points X0,X1 to input in the integral formula
					// This is to choose the inner point (X0)

					if (levelset[k1] > levelset[k0])
						X0 = support_points[k0];

					else X0 = support_points[k1];
					// Setting the points of the new face.

					new_face[intersected_face] = X1;
					intersected_face++;
					// Call new function to evaluate integral:
					// This function F represents the integration over this face.
					// This is added for each face to yield the final integration
					// over the cell (polygon)

					//The normal vector is the same for the new cut face.
					// face_normal_vector = fe_face_values.normal_vector(0);

					// This is to prevent the integration when the boundary
					// reaches a DOF such that a new face is created with
					// the same two DOF's.
					if (X0 != X1)
					{
						ObjNewCell_new.setCoordinates(integrate_face_check,X0,X1);
						ObjNewCell_new.setVertices();
						face_normal_vector = fe_face_values.normal_vector(0);

						// The only faces that are relevant to the stabilization term for the
						// usurface var. are these: the intersected faces.
						// For the ubulk var., I want to include the intersected AND inside faces.
						integrate_face_usurface = true;
						integrate_face_ubulk = true;
						new_normal_vector.push_back(face_normal_vector);
						new_face_vector.push_back(X0);
						new_face_vector.push_back(X1);

						new_face_vector.push_back(VOID_POINT);
						X0 = mapping.transform_real_to_unit_cell(cell,X0);
						X1 = mapping.transform_real_to_unit_cell(cell,X1);
						double dy = (X1[1] - X0[1]);
						double dx = (X1[0] - X0[0]);
						// Face Length has to be of the unit cell!
						// It is part of the mapping from the unit cell face/new cut face
						// to the parametric line r(t)
						double face_length = sqrt(dx*dx+dy*dy);
						for (unsigned int dof_i=0; dof_i<dofs_per_cell; ++dof_i) {
							for (unsigned int dof_j=0; dof_j<dofs_per_cell; ++dof_j) {
								dof_index_i = dof_handler_new.get_fe()[active_fe_index].
										system_to_component_index(dof_i).second;
								dof_index_j =
										dof_handler_new.get_fe()[active_fe_index].
										system_to_component_index(dof_j).second;
								if (dof_handler_new.get_fe()[active_fe_index].
										system_to_component_index(dof_i).first == 1 &&
										dof_handler_new.get_fe()[active_fe_index].
										system_to_component_index(dof_j).first == 1 ) {

								cell_matrix(dof_i,dof_j) += b_B*k_B*
										Obj_cut_cell_integration.return_face_integration
										(X0,X1,
												face_normal_vector,
												dof_index_i,dof_index_j, face_length); }
							}
							if (dof_handler_new.get_fe()[active_fe_index].
									system_to_component_index(dof_i).first == 1 ) {
								dof_index_i = dof_handler_new.get_fe()[active_fe_index].
										system_to_component_index(dof_i).second;
							cell_rhs(dof_i)+= f_B*
									Obj_cut_cell_integration.return_rhs_face_integration
									(X0,X1,face_normal_vector, dof_index_i, face_length);
							}
						}
						integrate_face_check++;
					}
				}
				// Do not integrate faces where one node is inside and the other is
				// a coincident node (this case is covered in the next if{} (inside faces)
				// The next 2 statements are only to include the coincident node as
				// a node for the new face. (It is in fact the intersection)
				// Regarding the stabilization term, only the stabilization matrix of the ubulk
				// term will include these faces; that is why integrate_face_usurface is not included.
				if (levelset[k0] == 0.0 && levelset[k1] < 0) {
					new_face[intersected_face] = support_points[k0];
					intersected_face++;
					integrate_face_usurface = true;
					integrate_face_ubulk = true;
				}
				else if (levelset[k1] == 0.0 && levelset[k0] < 0) {
					new_face[intersected_face] = support_points[k1];
					intersected_face++;
					integrate_face_usurface = true;
					integrate_face_ubulk = true;
				}
				// INTEGRATE INSIDE FACES, not intersected by the boundary.
				// Exclude faces with both DOFs outside the boundary.
				// TAKE CARe WITH ELSE IF!!!!!!!!!!!!!!!!!!!!!
				if (levelset[k0]<=0 && levelset[k1]<=0) {
					Point<dim> X0;
					Point<dim> X1;
					Point<dim> face_normal_vector;
					// changed: add face to stabilization matrix ONLY to the usurface var.
					integrate_face_ubulk = true;
					inside_faces++; // Keep track of number of inside_faces.
					X0 = support_points[k0];
					X1 = support_points[k1];
					ObjNewCell_new.setCoordinates(integrate_face_check,X0,X1);
					ObjNewCell_new.setVertices();
					new_face_vector.push_back(X0);
					new_face_vector.push_back(X1);
					new_face_vector.push_back(VOID_POINT);

					face_normal_vector = fe_face_values.normal_vector(1);

					X0 = mapping.transform_real_to_unit_cell(cell,X0);
					X1 = mapping.transform_real_to_unit_cell(cell,X1);

					new_normal_vector.push_back(face_normal_vector);
					double dy = (X1[1] - X0[1]);
					double dx = (X1[0] - X0[0]);

					double face_length = sqrt(dx*dx+dy*dy);

					for (unsigned int dof_i=0; dof_i<dofs_per_cell; ++dof_i) {
						for (unsigned int dof_j=0; dof_j<dofs_per_cell; ++dof_j) {

							dof_index_i = dof_handler_new.get_fe()[active_fe_index].
									system_to_component_index(dof_i).second;
							dof_index_j = dof_handler_new.get_fe()[active_fe_index].
									system_to_component_index(dof_j).second;
							if (dof_handler_new.get_fe()[active_fe_index].
									system_to_component_index(dof_i).first == 1 &&
									dof_handler_new.get_fe()[active_fe_index].
									system_to_component_index(dof_j).first == 1 )

							cell_matrix(dof_i,dof_j) += b_B*k_B*
									Obj_cut_cell_integration.return_face_integration
									(X0,X1,
											face_normal_vector,
											dof_index_i,dof_index_j, face_length);
						}
						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).first == 1){
							dof_index_i = dof_handler_new.get_fe()[active_fe_index].
									system_to_component_index(dof_i).second;
						cell_rhs(dof_i)+= f_B*
								Obj_cut_cell_integration.return_rhs_face_integration
								(X0,X1,face_normal_vector, dof_index_i, face_length);
						}
					}
					integrate_face_check++;
				}
				// NEW : Integrate term stabilization term j
				// j = sum over F belongs to Fg gamma1*h*
				// ( nf*grad_phi_i,K - nf*grad_phi_i,K`)*( nf*grad_phi_j,K - nf*grad_phi_j,K`)
				// Where F is a face of a cell belonging to the set of boundary cells
				// Fg. The face F is either an intersected face or an interior face (ie,
				// at least one node inside the boundary)

				// Find the neighbor cell K` that shares this face F with the present
				// cell K.
				// This will integrate all faces belonging to the "BULK" domain -
				// but I am interested in integrating The set "F_S,h - set of internal faces (i.e.,
				// faces with two neighbors)" According to "Cut FEM... Burman et al 2014"
				if (integrate_face_usurface)
				{
					for (unsigned int neighborCellIterator=0; neighborCellIterator
					< GeometryInfo<2>::faces_per_cell; ++neighborCellIterator)
					{

						if (cell->neighbor_index(neighborCellIterator) != -1) {

							neighbor_cell_global_dof_indices_ALL.resize(dofs_per_cell);
							neighbor_cell_global_dof_indices_ubulk.resize(dofs_per_cell);
							neighbor_face_dof_indices_ALL.resize(dofs_per_face);
							// Initialize FEValues for cell K'
//							fe_values_neighborCell.reinit
//							(cell->neighbor(neighborCellIterator));
							//	cell->face(face) // global face index
							for (unsigned int neighborCellFaceIterator=0;
									neighborCellFaceIterator < GeometryInfo<2> ::faces_per_cell;
									++neighborCellFaceIterator) {

									if (cell->neighbor(neighborCellIterator) // neighbor cell K'
											->face(neighborCellFaceIterator) // faces of K' cell
											== cell->face(face) )		     // face of
										// This means that the cell K' neighborCellIterator has a
										// face in common with the present cell K.
									{

										FEFaceValues<dim> &fe_face_values_neighborCell
										= fe_face_values_neighborCell_surface;

										fe_face_values_neighborCell.reinit(cell->neighbor(neighborCellIterator),
																	neighborCellFaceIterator);
										// The neighbor cell must be a surface cell as well; these condition
										// should be covered when integrate_face_usurface was activated.
										assert(cell_is_in_surface_domain(cell->neighbor(neighborCellIterator)));

										cell->neighbor(neighborCellIterator) // neighbor cell K'
												->get_dof_indices(neighbor_cell_global_dof_indices_ALL);
										// Create a neighbor_cell_global_dof_indices with only the indices
										// of the cell of usurface var.
										count_u = 0;
										for (unsigned int i=0; i<neighbor_cell_global_dof_indices_ALL.size(); ++i) {
											if (dof_handler_new.get_fe()[active_fe_index].
													system_to_component_index(i).first == 0) {
												neighbor_cell_global_dof_indices[count_u]
												           = neighbor_cell_global_dof_indices_ALL[i];
												count_u++;
											}
										}

										cell->neighbor(neighborCellIterator)
												->face(neighborCellFaceIterator)
												->get_dof_indices
												(neighbor_face_dof_indices_ALL,
														cell->neighbor(neighborCellIterator)->
														active_fe_index());
										count_u = 0;
										// Create a neighbor_face_dof_indices with only the indices
										// of the faces of the usurface var.
										for (unsigned int i=0; i<neighbor_face_dof_indices_ALL.size(); ++i) {
											if (dof_handler_new.get_fe()[active_fe_index].
													system_to_component_index(i).first == 0) {
												neighbor_face_dof_indices[count_u]
												                          = neighbor_face_dof_indices_ALL[i];
												count_u++;
											}
										}

										int face_global_index = cell->face_index(face);
										// Find if this face has already been submitted to the
										// j integration.
										// If this is true, face has not been found.
										if (std::find(j_integrate_face.begin(),
												j_integrate_face.end(), face_global_index)
										== j_integrate_face.end())
										{
											std::vector<int> EXTENDED_global_dof_indices;
											j_integrate_face.push_back(face_global_index);
											// Create a vector with the same global dof's as cell K...
											for (unsigned int i=0;i<dofs_per_cell/2;++i) {
												EXTENDED_global_dof_indices.push_back
												(cell_global_dof_indices[i]);
											}

											// Check which DOF's of cell K' don't belong to cell K and
											// add them to the extended vector.
											// EDITED: /2
											for (unsigned int i=0;i<dofs_per_cell/2;++i) {
												if (std::find(EXTENDED_global_dof_indices.begin(),
														EXTENDED_global_dof_indices.end(),
														neighbor_cell_global_dof_indices[i])
												== EXTENDED_global_dof_indices.end() )
												{
													EXTENDED_global_dof_indices.push_back
													( neighbor_cell_global_dof_indices[i] );
												}
											}

											// Save DOF's that were J_integrated to avoid
											// repeating
											for (unsigned int i=0;i<2;++i) {
												unsigned int j = face_dof_indices[i];
												j_face_vector.push_back(support_points[j]);

												if (std::find(j_face_vector_global_dofs.begin(),
														j_face_vector_global_dofs.end(),
														j) == j_face_vector_global_dofs.end())
												{
													j_face_vector_global_dofs.push_back(j);
												}
												else {
													// Avoid use repeated DOF's
													j_face_vector_global_dofs.push_back(VOID_INT);
												}
											}

											assert (face_dof_indices[0] == neighbor_face_dof_indices[0]);
											assert (face_dof_indices[1] == neighbor_face_dof_indices[1]);

											// Select the local DOF indices of the face in the neighbor cell K'
											// corresponding to the equivalent global DOF indices in the
											// face in cell cell K.
											std::vector<int> EXTENDED_local_dof_K;
											std::vector<int> EXTENDED_local_dof_K_neighbor;

											// EXTENDED_local_dof_K   = {0,1,2,3,-1,-1}
											// EXTENDED_local_dof_K_p = {1,3,5,7,-1,-1}
											// Change:
											// EXTENDED_local_dof_K   = {0,2,4,6,-1,-1}
											for (unsigned int i=0;i<dofs_per_cell/2;++i) {
//												EXTENDED_local_dof_K.push_back(i);
												EXTENDED_local_dof_K.push_back(2*i);
											}
											EXTENDED_local_dof_K.push_back(-1); // 4
											EXTENDED_local_dof_K.push_back(-1);	// 5
											for (unsigned int i=0;i<EXTENDED_global_dof_indices.size()
											;++i)
											{
												// Is global index (i) of neighbor cell K' a
												// global dof of cell K? If NOT:
												if (std::find(neighbor_cell_global_dof_indices.begin(),
														neighbor_cell_global_dof_indices.end(),
														EXTENDED_global_dof_indices[i])
												== neighbor_cell_global_dof_indices.end())
												{
													EXTENDED_local_dof_K_neighbor.push_back(-1);
												}
												else
												{
													for (unsigned int j=0;j<dofs_per_cell/2;++j){
														if (neighbor_cell_global_dof_indices[j] ==
																EXTENDED_global_dof_indices[i])
//															EXTENDED_local_dof_K_neighbor.push_back(j);
															EXTENDED_local_dof_K_neighbor.push_back(j*2);
													}
												}
											}
											calls_to_j++;
											// Important, never forget to reset local element matrix!
											cell_j_matrix = 0;
											for (unsigned int dof_i=0;dof_i<dofs_per_cell+4;++dof_i) {
												for (unsigned int dof_j=0;dof_j<dofs_per_cell+4;++dof_j) {

													if ( dof_i%2==0 && dof_j%2==0 ) {

														dof_index_i = ExtendedGlobal2Local(dof_i,cell);
														dof_index_j = ExtendedGlobal2Local(dof_j,cell);

														cell_j_matrix(dof_i, dof_j ) // dof_i,j = 0,2,4,6,(8,10)
														+=
																Obj_cut_cell_integration.
																getTermJ
																(fe_face_values,
																		fe_face_values_neighborCell,
																		dof_index_i,dof_index_j,
																		EXTENDED_local_dof_K,			 // K
																		EXTENDED_local_dof_K_neighbor,
																		usurface);  // K'
														// Inputing directly inside the global j-matrix also works:
														// NEW_j_matrix(EXTENDED_global_dof_indices[dof_i],
														// EXTENDED_global_dof_indices[dof_j])
													}
												} // end dof_i
											} // end dof_j
											for (unsigned int i=0; i<dofs_per_cell+4; ++i){
												for (unsigned int j=0; j<dofs_per_cell+4; ++j)
												{
													// if fe.system_to_comp... doesn't work because I have to
													// account for the extended dof's from the other cell.
													if ( i%2==0 && j%2==0 ) {
														dof_index_i = ExtendedGlobal2Local(i,cell);
														dof_index_j = ExtendedGlobal2Local(j,cell);
														j_matrix.add (EXTENDED_global_dof_indices[dof_index_i],
																EXTENDED_global_dof_indices[dof_index_j],
																cell_j_matrix(i,j));
													}
												}
											}
										} // end if face was found (I mean, if face was already j-integrated)
										j_face_vector.push_back(VOID_POINT);
										j_face_vector_global_dofs.push_back(VOID_INT);
										j_face_vector_global_dofs_p.push_back(VOID_INT);
									} // end if face is common
							} // end loop (neighbor) faces
						} // end if neighbor cell is !=-1
					} // end loop over neighbor cells
				} // end if integrate face stab term usurface
				if (integrate_face_ubulk) {
					for (unsigned int neighborCellIterator=0; neighborCellIterator
					< GeometryInfo<2>::faces_per_cell; ++neighborCellIterator)	{
						if (cell->neighbor_index(neighborCellIterator) != -1) {
							for (unsigned int neighborCellFaceIterator=0;
									neighborCellFaceIterator < GeometryInfo<2> ::faces_per_cell;
									++neighborCellFaceIterator)	{
								if (cell->neighbor(neighborCellIterator) // neighbor cell K'
										->face(neighborCellFaceIterator) // faces of K' cell
										== cell->face(face) ) {
									// This means that the cell K' neighborCellIterator has a
									// face in common with the present cell K.

									// If neighbor cell is inside the surface, it has 8 dofs per cell
									// and the FE objectis fe_face..._surface, associated with FESystem
									// fe_surface. If the cell is inside, it has 4 dofs and is associated
									// with fe_inside. It is a more complicated case because a coupling
									// between a cell with 8 dofs and a cell K' with 4 dofs is occurring.
									FEFaceValues<dim> &fe_face_values_neighborCell
									= (cell_is_in_surface_domain(cell->neighbor(neighborCellIterator))) ?
											fe_face_values_neighborCell_surface
											: fe_face_values_neighborCell_inside;

									fe_face_values_neighborCell.reinit(cell->neighbor(neighborCellIterator),
											neighborCellFaceIterator);

									int neighbor_dofs_per_cell
									= cell->neighbor(neighborCellIterator)->get_fe().dofs_per_cell;

									int neighbor_dofs_per_face
									= cell->neighbor(neighborCellIterator)->get_fe().dofs_per_face;

									neighbor_cell_global_dof_indices_ALL.resize(neighbor_dofs_per_cell);
									neighbor_cell_global_dof_indices_ubulk.resize(neighbor_dofs_per_cell);
									neighbor_face_dof_indices_ALL.resize(neighbor_dofs_per_face);

									// Get DOF indices of the neighbor CELL
									cell->neighbor(neighborCellIterator) // neighbor cell K'
													->get_dof_indices(neighbor_cell_global_dof_indices_ALL);
									count_p = 0;
									// If neighbor cell is in surface domain, cell has 8 dofs,
									// need to correct the neighbor dof count.
									if (cell_is_in_surface_domain(cell->neighbor(neighborCellIterator)))
										for (unsigned int i=0; i<neighbor_cell_global_dof_indices_ALL.size(); ++i) {
											if (dof_handler_new.get_fe()[active_fe_index].
													system_to_component_index(i).first == 1) {
												neighbor_cell_global_dof_indices_ubulk[count_p]
												                                       = neighbor_cell_global_dof_indices_ALL[i];
												count_p++;
											}
										}
									// if not, the get_dof_indices already got the right numbering
										else neighbor_cell_global_dof_indices_ubulk =
												neighbor_cell_global_dof_indices_ALL;
									// Get DOF indices of the neighbor FACE (is common to the K face)
										cell->neighbor(neighborCellIterator)
										->face(neighborCellFaceIterator)
										->get_dof_indices
										(neighbor_face_dof_indices_ALL,
										cell->neighbor(neighborCellIterator)->active_fe_index());

										count_p = 0;
										if (cell_is_in_surface_domain(cell->neighbor(neighborCellIterator)))
										for (unsigned int i=0; i<neighbor_face_dof_indices_ALL.size(); ++i) {
											if (dof_handler_new.get_fe()[active_fe_index].
													system_to_component_index(i).first == 1) {
												neighbor_face_dof_indices_ubulk[count_p]
												                            = neighbor_face_dof_indices_ALL[i];
												count_p++;
											}
										}
										else neighbor_face_dof_indices_ubulk = neighbor_face_dof_indices_ALL;

										int face_global_index = cell->face_index(face);
										// Find if this face has already been submitted to the
										// j integration.
										// If this is true, face has not been found.
										if (std::find(j_integrate_face_p.begin(),
												j_integrate_face_p.end(), face_global_index)
										== j_integrate_face_p.end())
										{
											std::vector<int> EXTENDED_global_dof_indices_p;
											j_integrate_face_p.push_back(face_global_index);
											// Create a vector with the same global dof's as cell K...
											for (unsigned int i=0;i<dofs_per_cell/2;++i) {
												EXTENDED_global_dof_indices_p.push_back
												(cell_global_dof_indices_ubulk[i]);
											}

											for (unsigned int i=0;i<dofs_per_cell/2;++i) {
												if (std::find(EXTENDED_global_dof_indices_p.begin(),
														EXTENDED_global_dof_indices_p.end(),
														neighbor_cell_global_dof_indices_ubulk[i])
												== EXTENDED_global_dof_indices_p.end() )
												{
													EXTENDED_global_dof_indices_p.push_back
														( neighbor_cell_global_dof_indices_ubulk[i] );
													}
												}
												assert(EXTENDED_global_dof_indices_p.size() == 6);

												for (unsigned int i=0;i<2;++i) {
													unsigned int j = face_dof_indices_ubulk[i];
													j_face_vector_usurface.push_back(support_points[j]);

													if (std::find(j_face_vector_global_dofs_p.begin(),
															j_face_vector_global_dofs_p.end(),
															j) == j_face_vector_global_dofs_p.end())
													{
														j_face_vector_global_dofs_p.push_back(j);
													}
													else {
														// Avoid use repeated DOF's
														j_face_vector_global_dofs_p.push_back(VOID_INT);
													}
												}

												assert (face_dof_indices_ubulk[0] == neighbor_face_dof_indices_ubulk[0]);
												assert (face_dof_indices_ubulk[1] == neighbor_face_dof_indices_ubulk[1]);

												// Select the local DOF indices of the face in the neighbor cell K'
												// corresponding to the equivalent global DOF indices in the
												// face in cell cell K.
												std::vector<int> EXTENDED_local_dof_K_p;
												std::vector<int> EXTENDED_local_dof_K_neighbor_p;

												// EXTENDED_local_dof_K_p = {1,3,5,7,-1,-1}
												for (unsigned int i=1;i<dofs_per_cell/*/2*/; ++(++i)) {
													EXTENDED_local_dof_K_p.push_back(i);
												}
												EXTENDED_local_dof_K_p.push_back(-1); 	// 4
												EXTENDED_local_dof_K_p.push_back(-1);	// 5

												// Create an extended "unit" neighbor cell, with extra dofs representing the
												// face that is not common between cell K and K': in here, represented by
												// setting the dof to -1.

												for (unsigned int i=0;i<EXTENDED_global_dof_indices_p.size()
												;++i)
												{
													// Is global index (i) of neighbor cell K' a
													// global dof of cell K? If NOT:
													if (std::find(neighbor_cell_global_dof_indices_ubulk.begin(),
															neighbor_cell_global_dof_indices_ubulk.end(),
															EXTENDED_global_dof_indices_p[i])
													== neighbor_cell_global_dof_indices_ubulk.end()) {
														EXTENDED_local_dof_K_neighbor_p.push_back(-1);
													}
													// If global DOF of K' is a global DOF of K (same face):
													else {

														for (unsigned int j=0;j<dofs_per_cell/2;++j){
															if (neighbor_cell_global_dof_indices_ubulk[j] == EXTENDED_global_dof_indices_p[i]){
																if (cell_is_in_surface_domain(cell->neighbor(neighborCellIterator)))
																	// If K' is in surface, the unit cell dof will be on the form
																	// 1,3,5,7,-1,-1 - that's why j*2+1.
																	// These dof's need to be set like this because in the
																	// getTermJ call, the calculation is done via fe_values[extractor]
																	// where the fe_values object expects a 8 DOF's cell.
																	EXTENDED_local_dof_K_neighbor_p.push_back(j*2+1);
																// If K' is inside, the unit cell is the traditional (0,1,2,3,-1,-1)
																// A different getTermJ is called, where the fe_values is single (not fe_system)
																else EXTENDED_local_dof_K_neighbor_p.push_back(j);
															}
														}
													}
												}
												// calls_to_j++;
												// Important, never forget to reset local element matrix!
												// matrix of J term is always referring to the surface cell, which has
												// dofs_per_cell+4 terms.
												cell_j_matrix_p.reinit(dofs_per_cell+4,dofs_per_cell+4);
												cell_j_matrix_p = 0;
												for (unsigned int dof_i=0;dof_i<dofs_per_cell+4;++dof_i) {
													for (unsigned int dof_j=0;dof_j<dofs_per_cell+4;++dof_j) {

														if ( dof_i%2!=0 && dof_j%2!=0 ) {

															dof_index_i = ExtendedGlobal2Local(dof_i,cell);
															dof_index_j = ExtendedGlobal2Local(dof_j,cell);

															if (cell_is_in_surface_domain(cell->neighbor(neighborCellIterator)))
																cell_j_matrix_p(dof_i, dof_j )
																// dof_i,j = 0,2,4,6,(8,10)
																+=      Obj_cut_cell_integration.
																getTermJ
																(fe_face_values,
																		fe_face_values_neighborCell,
																		dof_index_i,dof_index_j,
																		EXTENDED_local_dof_K_p,			 // K
																		EXTENDED_local_dof_K_neighbor_p,
																		ubulk);  // K'
															// getTermJ_p_inside function is adapted for a fe_face_values_neighbor object
															// containing 2 variables (it is derived from fe_surface)
															else
																cell_j_matrix_p(dof_i, dof_j )
																// dof_i,j = 0,2,4,6,(8,10)
																+= Obj_cut_cell_integration.
																getTermJ_mixed(fe_face_values,
																		fe_face_values_neighborCell,
																		dof_index_i,dof_index_j,
																		EXTENDED_local_dof_K_p,			 // K
																		EXTENDED_local_dof_K_neighbor_p);  // K'

														} // end if dof is odd
													} // end dof_i
												} // end dof_j
												for (unsigned int i=0; i<dofs_per_cell+4; ++i)
													for (unsigned int j=0; j<dofs_per_cell+4; ++j){
														if ( i%2!=0 && j%2!=0 ) {
															dof_index_i = ExtendedGlobal2Local(i,cell);
															dof_index_j = ExtendedGlobal2Local(j,cell);
															j_matrix_p.add
															(EXTENDED_global_dof_indices_p[dof_index_i] - n_dofs_surface,
																		EXTENDED_global_dof_indices_p[dof_index_j]- n_dofs_surface,
																		cell_j_matrix_p(i,j));
															}
														}

												// END SAME FOR P/* */
												j_face_vector_usurface.push_back(VOID_POINT);
										} // end if face was found (I mean, if face was already j-integrated)
										j_face_vector.push_back(VOID_POINT);
										j_face_vector_global_dofs.push_back(VOID_INT);
										j_face_vector_global_dofs_p.push_back(VOID_INT);

									} // end if face is common
									//} // end if neighbor cell has fe_index == 0 (ie,is a boundary cell)
								} // end loop (neighbor) faces
							} // end if neighbor cell is !=-1
//						} // end if neighbor cell is bulk
					} // end loop over neighbor cells
				}// end integrate face stab term ubulk (p)

			} // End loop faces

			// Make sure that all the inside and intersected faces that were integrated
			// before are now integrated by the J term. (doesn't include new cut faces)
//			assert(calls_to_j == integrate_face_check);
			assert(integrate_face_check >= 2);

			//INTEGRATE NEW FACE (BOUNDARY)
			// Now, one must integrate the new face created by the boundary on the cell.
			// The two intersection points found in the previous loops will create the
			// new face with the following points:
			Point<dim> X0;
			Point<dim> X1;
			Point<dim> face_normal_vector;

			X0 = new_face[0];
			X1 = new_face[1];
			ObjNewCell_new.setCoordinates(integrate_face_check,X0,X1);
			ObjNewCell_new.setVertices();
			new_face_vector.push_back(X0);
			new_face_vector.push_back(X1);

			create_levelset_vector_map(levelset_face_map,X0);
			create_levelset_vector_map(levelset_face_map,X1);

			new_face_vector.push_back(VOID_POINT);
			face_normal_vector = ObjNewCell_new.getCutFaceNormal();
			new_normal_vector.push_back(face_normal_vector);

			// Note that the terms regarding the integration over the entire cell
			// a = (grad phi_i, grad phi_j) and b = (f,phi_i) use the UNIT face_length (FIXED),
			// whereas the terms of integration over the boundary uses the REAL real_face_length (fixed)
			double real_face_length =
					ObjNewCell_new.ObjNewFace[integrate_face_check].face_length;

			X0 = mapping.transform_real_to_unit_cell(cell,X0);
			X1 = mapping.transform_real_to_unit_cell(cell,X1);

			double face_length = ObjNewCell_new.distance(X0,X1);
			for (unsigned int dof_i=0; dof_i<dofs_per_cell; ++dof_i) {
				for (unsigned int dof_j=0; dof_j<dofs_per_cell; ++dof_j) {

					dof_index_i = dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_i).second;
					dof_index_j = dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_j).second;
					if (dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_i).first == 1 &&
							dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_j).first == 1 )

						cell_matrix(dof_i,dof_j) += b_B*k_B*
								Obj_cut_cell_integration.return_face_integration
								(X0,X1,
										face_normal_vector,
										dof_index_i,dof_index_j, face_length);
					// Integrate boundary term (L-B operator) : this is already inside
					// only boundary cells.
					if (dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_i).first == 0 &&
							dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_j).first == 0 ) {
						cell_matrix(dof_i,dof_j) += b_S*k_S*
								Obj_cut_cell_integration.getTermBeltramiBoundary
								(X0,X1,	face_normal_vector,	dof_index_i,
										dof_index_j,
										real_face_length
										);
					}
				} // End For dof_j
				// Integrate COUPLING term // moved from here
				// Integrate RHS bulk term
				if (dof_handler_new.get_fe()[active_fe_index].
						system_to_component_index(dof_i).first == 1) {
					dof_index_i = dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_i).second;
					cell_rhs(dof_i)+= f_B*
							Obj_cut_cell_integration.return_rhs_face_integration
							(X0,X1,face_normal_vector, dof_index_i, face_length)
							// Abort this term for the evaluation of coupled system
							// (these terms do not appear on "Cut FEM... Burman et al 2014")
							// (If g_D = 0, this doesn't change anything actually...)
//							+
//							Obj_cut_cell_integration.getTermD2
//							(X0, X1, face_normal_vector,
//									dof_index_i, alfa, g_D, real_face_length )
									;
				}
				// Integrate RHS boundary term (L-B operator)
				if (dof_handler_new.get_fe()[active_fe_index].
						system_to_component_index(dof_i).first == 0) {
					dof_index_i = dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_i).second;
//					 original: fs = 108*sin...
//					fs = 108*sin(3*atan2
					fs = 108/12*sin(3*atan2
							(support_points[cell_global_dof_indices_ubulk[dof_index_i]][1],
									support_points[cell_global_dof_indices_ubulk[dof_index_i]][0]));
					cell_rhs(dof_i)+=
							b_S*Obj_cut_cell_integration.getTermBeltramiBoundaryRHS
							(X0,X1,dof_index_i,
									real_face_length
									, fs);

					constraint_vector (cell_global_dof_indices_ALL[dof_i])+=
							Obj_cut_cell_integration.constraintVector
							(X0,X1, dof_index_i,
//									face_length
									real_face_length
									);
				}
			}  // End For dof_i

			// Integrate COUPLING term // moved to here
			int corrector_u_i, corrector_u_j;
			int corrector_p_i, corrector_p_j;
			if(0){
				for (unsigned int dof_i=0; dof_i<dofs_per_cell; ++dof_i) {
					for (unsigned int dof_j=0; dof_j<dofs_per_cell; ++dof_j) {

						dof_index_i = dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).second;
						dof_index_j = dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_j).second;


						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).first == 0)
							corrector_u_i = 1;
						else corrector_u_i = 0;

						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_j).first == 0)
							corrector_u_j = 1;
						else corrector_u_j = 0;

						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).first == 1)
							corrector_p_i = 1;
						else corrector_p_i = 0;

						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_j).first == 1)
							corrector_p_j = 1;
						else corrector_p_j = 0;

						cell_matrix(dof_i,dof_j) +=
								Obj_cut_cell_integration.getTermCoupling
								(X0,X1,	face_normal_vector,	dof_index_i,
										dof_index_j,
										real_face_length,
										corrector_u_i,
										corrector_u_j,
										corrector_p_i,
										corrector_p_j,
										b_B,b_S);
					}
				}
			}

			// Abort these terms for the evaluation of coupled system (these terms do not appear
			// on "Cut FEM... Burman et al 2014")
			// TERM C
			if(1)
			{
				for (unsigned int dof_i=0; dof_i<dofs_per_cell; ++dof_i) {
					for (unsigned int dof_j=0; dof_j<dofs_per_cell; ++dof_j) {
						dof_index_i = dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).second;
						dof_index_j = dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_j).second;
						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).first == 1 &&
								dof_handler_new.get_fe()[active_fe_index].
							system_to_component_index(dof_j).first == 1 )

							cell_matrix(dof_i,dof_j) +=
									Obj_cut_cell_integration.getTermC(X0, X1, real_face_length,
											face_normal_vector,dof_index_i,dof_index_j);
					} // End dof_i
				}// End dof_j

				// TERM D
				for (unsigned int dof_i=0; dof_i<dofs_per_cell; ++dof_i) {
					for (unsigned int dof_j=0; dof_j<dofs_per_cell; ++dof_j) {
						dof_index_i = dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).second;
						dof_index_j = dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_j).second;
						if (dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_i).first == 1 &&
								dof_handler_new.get_fe()[active_fe_index].
								system_to_component_index(dof_j).first == 1 )
							cell_matrix(dof_i,dof_j) +=
									Obj_cut_cell_integration.getTermD(X0, X1, real_face_length,
											dof_index_i,dof_index_j,alfa);
					} // End dof_i
				} // End dof_j
			} // end if use term C,D

//			ExactSolutionUsurface<dim> extract_exact_solutionUsurface;
			// This will make the values outside the boundary go to zero:
			// "weight" the DOF's outside the solution boundary, so that every DOF outside the boundary
			// reports a value of 0 for the exact solution, not following the analytic solution.
			// This is done again in the error evaluation, where the value of the DOF's outside the boundary
			// are weighted to zero. Therefore, this serves only to visualize the solution better.
//			WeightSolutionUbulk<dim> extract_weight_exact_solution;
			// Update: The weighting is not needed anymore for the exact solution, since I
			// eliminated the cells outside the boundary. BUT, as I intend to keep the outside
			// triangulation in the future and work with one more FENothing, this will be kept.


			//Creating the vector of exact solution:
			for (unsigned int i=0; i<dofs_per_cell; ++i) {
				// Create exact_solution vector for usurface
				if (dof_handler_new.get_fe()[active_fe_index].
						system_to_component_index(i).first == 0) {
				unsigned int j = cell_global_dof_indices_ALL[i];
				exact_solution_usurface[j] =  extract_exact_solutionUsurface.value(support_points[j])
//					 *extract_weight_exact_solution.value(support_points[j])
					 ;
				}
				// Create exact_solution vector for ubulk
				else {
					unsigned int j = cell_global_dof_indices_ALL[i];
					exact_solution_ubulk[j-n_dofs_surface] =  extract_exact_solutionUbulk.value(support_points[j])
//						 *extract_weight_exact_solution.value(support_points[j])
						 ;
					}
			}
		} // End if (cell_is_in_surface_domain (isboundary))
		cell->get_dof_indices (local_dof_indices);
		// local_dof_indices is the vector with the global dof indices associated with this cell.
		for (unsigned int i=0; i<dofs_per_cell; ++i)
			for (unsigned int j=0; j<dofs_per_cell; ++j)
				system_matrix_new.add (local_dof_indices[i],
						local_dof_indices[j],
						cell_matrix(i,j));
		for (unsigned int i=0; i<dofs_per_cell; ++i)
			system_rhs_new(local_dof_indices[i]) += cell_rhs(i);
		count_cell++;

	} // (end for cell)

	FM_system_matrix.copy_from(system_matrix_new);
	// Variational Formulation from Burman and Hansbo "A unfitted..." (Laplace Problem with Nitsche's)

	FM_system_matrix_block00.copy_from(system_matrix_new.block(0,0));

	// Create block 11 without stabilization term j
	FM_system_matrix_block11.copy_from(system_matrix_new.block(1,1));
	// Create block 11 with stabilization term j
	FM_system_matrix_block11_yes_j.copy_from(system_matrix_new.block(1,1));
	// Formulation by Burman et al 2014 (Cut FEM...)
//	FM_system_matrix_block11_yes_j.add(cell_diameter*cell_diameter*cell_diameter*gamma_p,j_matrix_p);
	//Original formulation (An unfitted... Hansbo, Hansbo 2002);
	// used only to compare the errors with previous solution
	FM_system_matrix_block11_yes_j.add(cell_diameter*gamma_p,j_matrix_p);


	// Create block 00 with stab. term as described in "Cut FEM... " Burman et al 2014.
	FM_system_matrix_block00_w_j.copy_from(system_matrix_new.block(0,0));
	FM_system_matrix_block00_w_j.add(gamma_1,j_matrix);

	// Create matrices (block 00, surface) with constraint and with / without stabilization.
	// These matrices are augmented and have one more line and column.
	FM_system_matrix_block00_no_j_yes_constraint.reinit (n_dofs_surface+1,n_dofs_surface+1);
	FM_system_matrix_block00_yes_j_yes_constraint.reinit (n_dofs_surface+1,n_dofs_surface+1);

	for(unsigned int i = 0; i < n_dofs_surface; ++i) {
		for(unsigned int j = 0; j < n_dofs_surface; ++j) {

			FM_system_matrix_block00_no_j_yes_constraint(i,j) = FM_system_matrix(i,j);
			FM_system_matrix_block00_yes_j_yes_constraint(i,j)
					= FM_system_matrix(i,j) + gamma_1*j_matrix(i,j);
		}
	}
	// Add constraint vector to the last line
	// Note! The size of the matrix is n_dofs_surface+1, but the index of the last line is
	// n_dofs_surface (not this+1).
	// These constrained matrices will only serve when solving the problem decoupled; to solve the
	// full augmented matrix, I will input the constraint vector to the last column and vector of
	// the matrix AGAIN

	for(int unsigned j = 0; j < n_dofs_surface; ++j) {
		unsigned int i = n_dofs_surface;
		FM_system_matrix_block00_yes_j_yes_constraint(i,j) = constraint_vector.block(0)(j);
		FM_system_matrix_block00_no_j_yes_constraint (i,j) = constraint_vector.block(0)(j);
	}

	for(int unsigned i = 0; i < n_dofs_surface; ++i) {
		unsigned int j = n_dofs_surface;
		FM_system_matrix_block00_yes_j_yes_constraint(i,j) = constraint_vector.block(0)(i);
		FM_system_matrix_block00_no_j_yes_constraint(i,j)  = constraint_vector.block(0)(i);
	}

	// Create the "global" matrix that include the block 00 with stabilization and block 11
	// with stabilization. This matrix will yield the final solution to the coupling
	// problem. Constraint of the block 00 will be added to the end.
	// Input block 00

	FM_system_matrix_block_w_constraint.reinit(dof_handler_new.n_dofs()+1,dof_handler_new.n_dofs()+1);
	for(int unsigned i = 0; i < n_dofs_surface; ++i) {
		for(int unsigned j = 0; j < n_dofs_surface; ++j) {
			FM_system_matrix_block_w_constraint(i,j) = FM_system_matrix_block00_w_j(i,j);
		}
	}
	// Input block 11
	for(int unsigned i = 0; i < n_dofs_inside; ++i) {
		for(int unsigned j = 0; j < n_dofs_inside; ++j) {
			FM_system_matrix_block_w_constraint	(i+ n_dofs_surface,j+n_dofs_surface)
			= FM_system_matrix_block11_yes_j(i,j);
		}
	}
//	Input Block 01
	for(int unsigned i = 0; i < n_dofs_surface; ++i) {
		for(int unsigned j = n_dofs_surface; j < dof_handler_new.n_dofs(); ++j) {
			FM_system_matrix_block_w_constraint(i,j) = FM_system_matrix(i,j);
		}
	}
	// Input block 10
	for(int unsigned i = n_dofs_surface; i < dof_handler_new.n_dofs(); ++i) {
		for(int unsigned j = 0; j < n_dofs_surface; ++j) {
			FM_system_matrix_block_w_constraint	(i,j) = FM_system_matrix(i,j);
		}
	}

//	input constraints

	for(int unsigned j = 0; j < n_dofs_surface; ++j) {
		unsigned int i = dof_handler_new.n_dofs();
		FM_system_matrix_block_w_constraint(i,j) = constraint_vector.block(0)(j);
	}
	for(int unsigned i = 0; i < n_dofs_surface; ++i) {
		unsigned int j = dof_handler_new.n_dofs();
		FM_system_matrix_block_w_constraint(i,j) = constraint_vector.block(0)(i);
	}

	// Create the "global" matrix that include block 00 with stabilization and the NON augmented block 11
	// with stabilization and withOUT constraint. This matrix will yield the final solution
	// to the coupling problem.
	// Input block 00

	FM_system_matrix_block_yes_j_no_constraint.reinit(dof_handler_new.n_dofs()
			,dof_handler_new.n_dofs());
	for(int unsigned i = 0; i < n_dofs_surface; ++i) {
		for(int unsigned j = 0; j < n_dofs_surface; ++j) {
			FM_system_matrix_block_yes_j_no_constraint(i,j) = FM_system_matrix_block00_w_j(i,j);
		}
	}
	// Input block 11
	for(int unsigned i = 0; i < n_dofs_inside; ++i) {
		for(int unsigned j = 0; j < n_dofs_inside; ++j) {
			FM_system_matrix_block_yes_j_no_constraint	(i+ n_dofs_surface,j+n_dofs_surface)
			= FM_system_matrix_block11_yes_j(i,j);
		}
	}

	//	Input Block 01
		for(int unsigned i = 0; i < n_dofs_surface; ++i) {
			for(int unsigned j = n_dofs_surface; j < dof_handler_new.n_dofs(); ++j) {
				FM_system_matrix_block_yes_j_no_constraint(i,j) = FM_system_matrix(i,j);
			}
		}
		// Input block 10
		for(int unsigned i = n_dofs_surface; i < dof_handler_new.n_dofs(); ++i) {
			for(int unsigned j = 0; j < n_dofs_surface; ++j) {
				FM_system_matrix_block_yes_j_no_constraint	(i,j) = FM_system_matrix(i,j);
			}
		}

	// Calculate condition number
	// Do you really want to calculate the condition number?
	// It takes some time to invert the last matrices.
	double condition_number = 1;
	FullMatrix<double> IFM_system_matrix;

	// Calculate the Condition Number of the ENTIRE matrix, with constraint and without (both with J)
	if (1) {
	IFM_system_matrix.copy_from(FM_system_matrix_block_w_constraint);
//	IFM_system_matrix.copy_from(FM_system_matrix_block_yes_j_no_constraint);
	condition_number = IFM_system_matrix.l1_norm()*FM_system_matrix_block_w_constraint.l1_norm();
	std::cout << "L1 Condition Number w constraint.: " <<  condition_number << "\n";
	}

	if (0) {
//		IFM_system_matrix.copy_from(FM_system_matrix_block_w_constraint);
		IFM_system_matrix.copy_from(FM_system_matrix_block_yes_j_no_constraint);
		condition_number = IFM_system_matrix.l1_norm()*FM_system_matrix_block_yes_j_no_constraint.l1_norm();
		std::cout << "L1 Condition Number wOUT constraint.: " <<  condition_number << "\n";
		}

	// Calculating the Condition Number of the block 0,0 with J matrix.
	// It works well with and without adding Terms C and D: 7460.39 without them; 54437.6 with them.
	if (0) {
	IFM_system_matrix.copy_from(FM_system_matrix_block00_w_j);
	IFM_system_matrix.gauss_jordan();
//	condition_number = IFM_system_matrix.l1_norm()*FM_system_matrix_block00_w_j.l1_norm();
	std::cout << "L1 Condition Number block 00 w/out constraint.: "
			<<  IFM_system_matrix.l1_norm()*FM_system_matrix_block00_w_j.l1_norm() << "\n";
	}

	// Calculating the Condition Number of the block 1,1 with J matrix without and with constraint.
	// BOTH are singular. May this be because of all the elements that are zero inside the bulk mesh?
	// (The elements that are exclusively from the var. UBULK
	if (0) {
	IFM_system_matrix.copy_from(FM_system_matrix_block00_yes_j_yes_constraint);
	IFM_system_matrix.gauss_jordan();
	condition_number = IFM_system_matrix.l1_norm()*FM_system_matrix_block00_yes_j_yes_constraint.l1_norm();
	std::cout << "L1 Condition Number block 00 w/constraint, stabilization: " <<  condition_number << "\n";
	}

	ALLERRORS_usurface[cycle][7] = condition_number;
	// Create system_rhs_block0
	system_rhs_block0.reinit(system_rhs_new.block(0).size());
	system_rhs_block0 = system_rhs_new.block(0);
	// Create system_rhs_block1
	system_rhs_block1.reinit(system_rhs_new.block(1).size());
	system_rhs_block1 = system_rhs_new.block(1);

	// Create augmented rhs block 0 with constraints (new element = 0).
	system_rhs_block0_w_constraint.reinit(n_dofs_surface+1);
	for(unsigned int i = 0; i < n_dofs_surface; ++i) {
		system_rhs_block0_w_constraint(i) = system_rhs_block0(i);
	}
	system_rhs_block0_w_constraint(n_dofs_surface) = 0;
	// Create augmented RHS block 0 and 1 with constraints.
	system_rhs_block01_w_constraint.reinit(dof_handler_new.n_dofs()+1);
	for(unsigned int i = 0; i < n_dofs_surface; ++i) {
		system_rhs_block01_w_constraint(i) = system_rhs_block0(i);
	}
	for(unsigned int i = 0; i < n_dofs_inside; ++i) {
		system_rhs_block01_w_constraint(i+n_dofs_surface)= system_rhs_block1(i);
	}
	system_rhs_block01_w_constraint(dof_handler_new.n_dofs()) = 0;
	// Output matrices and vectors for visualization purposes only.
	// Output also vectors used to create plots in Gnuplot
	if(1) {


		// Output block 0,0 without stabilization.
//		{
//			std::ofstream MATRIX;
//			MATRIX.open("FM_system_matrix_block00_no_j.txt");
//			for(int unsigned i = 0; i < FM_system_matrix_block00.size(0); ++i) {
//				for(int unsigned j = 0; j < FM_system_matrix_block00.size(1); ++j) {
//					MATRIX << FM_system_matrix_block00(i,j) << ',';
//				}
//				MATRIX << std::endl;
//			}
//			MATRIX.close();
//		}
//
//		// Output block 0,0 with stabilization.
//		{
//			std::ofstream MATRIX;
//			MATRIX.open("FM_system_matrix_block00_w_j.txt");
//			for(int unsigned i = 0; i < FM_system_matrix_block00_w_j.size(0); ++i) {
//				for(int unsigned j = 0; j < FM_system_matrix_block00_w_j.size(1); ++j) {
//					MATRIX << FM_system_matrix_block00_w_j(i,j) << ',';
//				}
//				MATRIX << std::endl;
//			}
//			MATRIX.close();
//		}
//
		// Output the (virgin) stiffness matrix / or inverted into a .txt file. I can use the matrix in
		// Matlab and do relevant operations (For example, find the l2 norm and l2 cond.
		// number, which is not straight forward in deal.ii)
		// Output the complete system matrix, without stabilization and constraints.
		{
			std::ofstream FM_SYSTEM_MATRIX;
			FM_SYSTEM_MATRIX.open("FM_SYSTEM_MATRIX_COMPLETE_no_j_no_constraint.txt");
			for(int unsigned i = 0; i < FM_system_matrix.size(0); ++i) {
				for(int unsigned j = 0; j < FM_system_matrix.size(1); ++j) {
					FM_SYSTEM_MATRIX << FM_system_matrix(i,j) << ',';
				}
				FM_SYSTEM_MATRIX << std::endl;
			}
			FM_SYSTEM_MATRIX.close();
		}

		// Output the complete system matrix, with stabilization and constraints, coupling term.
		{
			std::ofstream FM_SYSTEM_MATRIX;
			FM_SYSTEM_MATRIX.open("FM_system_matrix_block_w_constraint_yes_j_yes_constraint.txt");
			for(int unsigned i = 0; i < FM_system_matrix_block_w_constraint.size(0); ++i) {
				for(int unsigned j = 0; j < FM_system_matrix_block_w_constraint.size(1); ++j) {
					FM_SYSTEM_MATRIX << FM_system_matrix_block_w_constraint(i,j) << ',';
				}
				FM_SYSTEM_MATRIX << std::endl;
			}
			FM_SYSTEM_MATRIX.close();
		}
		// Output the complete system matrix, with stabilization and NO constraints, coupling term.

		{
			std::ofstream FM_SYSTEM_MATRIX;
			FM_SYSTEM_MATRIX.open("FM_system_matrix_block_yes_j_no_constraint.txt");
			for(int unsigned i = 0; i < FM_system_matrix_block_yes_j_no_constraint.size(0); ++i) {
				for(int unsigned j = 0; j < FM_system_matrix_block_yes_j_no_constraint.size(1); ++j) {
					FM_SYSTEM_MATRIX << FM_system_matrix_block_yes_j_no_constraint(i,j) << ',';
				}
				FM_SYSTEM_MATRIX << std::endl;
			}
			FM_SYSTEM_MATRIX.close();
		}
//
//
//		// Output block 11 without j, constraint.
//		{
//			std::ofstream MATRIX;
//			MATRIX.open("FM_system_matrix_block11_no_j_no_constraint.txt");
//			for(int unsigned i = 0; i < FM_system_matrix_block11.size(0); ++i) {
//				for(int unsigned j = 0; j < FM_system_matrix_block11.size(1); ++j) {
//					MATRIX << FM_system_matrix_block11(i,j) << ',';
//				}
//				MATRIX << std::endl;
//			}
//			MATRIX.close();
//		}
//
//		// Create block11 and add the j_matrix_p to it.
//
//
//		// output block 11 with stabilization.
//		{
//			std::ofstream MATRIX;
//			MATRIX.open("FM_system_matrix_block11_yes_j.txt");
//			for(int unsigned i = 0; i < FM_system_matrix_block11_yes_j.size(0); ++i) {
//				for(int unsigned j = 0; j < FM_system_matrix_block11_yes_j.size(1); ++j) {
//					MATRIX << FM_system_matrix_block11_yes_j(i,j) << ',';
//				}
//				MATRIX << std::endl;
//			}
//			MATRIX.close();
//		}
//
//
//
//		// Output block 11 with j with constraints.
//		{
//			std::ofstream MATRIX;
//			MATRIX.open("FM_system_matrix_block00_yes_j_yes_constraint.txt");
//			for(int unsigned i = 0; i < FM_system_matrix_block00_yes_j_yes_constraint.size(0); ++i) {
//				for(int unsigned j = 0; j < FM_system_matrix_block00_yes_j_yes_constraint.size(1); ++j) {
//					MATRIX << FM_system_matrix_block00_yes_j_yes_constraint(i,j) << ',';
//				}
//				MATRIX << std::endl;
//			}
//			MATRIX.close();
//		}
//		// Output block 11 without J with constraints.
//		{
//			std::ofstream MATRIX;
//			MATRIX.open("FM_system_matrix_block00_no_j_yes_constraint.txt");
//			for(int unsigned i = 0; i < FM_system_matrix_block00_no_j_yes_constraint.size(0); ++i) {
//				for(int unsigned j = 0; j < FM_system_matrix_block00_no_j_yes_constraint.size(1); ++j) {
//					MATRIX << FM_system_matrix_block00_no_j_yes_constraint(i,j) << ',';
//				}
//				MATRIX << std::endl;
//			}
//			MATRIX.close();
//		}
//
//		// Output RHS block 0 1 without constraint.
//		{
//			std::ofstream SYSTEM_RHS;
//			SYSTEM_RHS.open("system_rhs_complete_no_constraint.txt");
//			for(int unsigned i = 0; i < system_rhs_new.size(); ++i) {
//				SYSTEM_RHS << system_rhs_new(i);
//				SYSTEM_RHS << std::endl;
//			}
//			SYSTEM_RHS.close();
//		}
//
//		// Output system rhs block 0
//		{
//			std::ofstream SYSTEM_RHS;
//			SYSTEM_RHS.open("system_rhs_block0.txt");
//			for(int unsigned i = 0; i < system_rhs_block0.size(); ++i) {
//				SYSTEM_RHS << system_rhs_block0(i);
//				SYSTEM_RHS << std::endl;
//			}
//			SYSTEM_RHS.close();
//		}
//		// Output system rhs block 1 without constraint.
//		{
//			std::ofstream SYSTEM_RHS;
//			SYSTEM_RHS.open("system_rhs_block1_no_constraint.txt");
//			for(int unsigned i = 0; i < system_rhs_block1.size(); ++i) {
//				SYSTEM_RHS << system_rhs_block1(i);
//				SYSTEM_RHS << std::endl;
//			}
//			SYSTEM_RHS.close();
//		}
//
//		// Output system rhs block 1 with constraint.
//		{
//			std::ofstream SYSTEM_RHS;
//			SYSTEM_RHS.open("system_rhs_block0_w_constraint.txt");
//			for(int unsigned i = 0; i < system_rhs_block0_w_constraint.size(); ++i) {
//				SYSTEM_RHS << system_rhs_block0_w_constraint(i);
//				SYSTEM_RHS << std::endl;
//			}
//			SYSTEM_RHS.close();
//		}
//
		// Output constraint vector.
		{
			std::ofstream CONSTRAINT_VECTOR;
			CONSTRAINT_VECTOR.open("constraint_vector.txt");
			for(int unsigned i = 0; i < constraint_vector.size(); ++i) {
				CONSTRAINT_VECTOR << constraint_vector(i);
				CONSTRAINT_VECTOR << std::endl;
			}
			CONSTRAINT_VECTOR.close();
		}
//
//		// Visualize the stabilization matrix for the ubulk variable
//		{
//			std::ofstream J_MATRIX_P;
//			J_MATRIX_P.open("j_matrix_p.txt");
//			for(int unsigned i = 0; i < j_matrix_p.size(0); ++i) {
//				for(int unsigned j = 0; j < j_matrix_p.size(1); ++j) {
//					J_MATRIX_P << j_matrix_p(i,j) << ',';
//				}
//				J_MATRIX_P << std::endl;
//			}
//			J_MATRIX_P.close();
//		}
		// Visualize the stabilization matrix for the usurface variable
//		{
//			std::ofstream J_MATRIX_U;
//			J_MATRIX_U.open("j_matrix_u.txt");
//			for(int unsigned i = 0; i < j_matrix.size(0); ++i) {
//				for(int unsigned j = 0; j < j_matrix.size(1); ++j) {
//					J_MATRIX_U << j_matrix(i,j) << ',';
//				}
//				J_MATRIX_U << std::endl;
//			}
//			J_MATRIX_U.close();
//		}
//
		// Start Output of vectors for visualization of relevant mesh in Gnuplot
		{
			// Save a .txt file with the coordinates of the faces relevant to the
			// stabilization term (j_faces)
			std::ofstream j_faces;
			j_faces.open("j_faces.txt");
			for(int unsigned j = 0; j <  j_face_vector.size(); ++j) {
				if (j_face_vector[j] == VOID_POINT) {
					j_faces << std::endl;
				}
				else {
					// If you want to output the global DOF's to visualize them over the nodes, uncomment
					// the lines below. Note that it will not be possible to "join" the points together
					// in the output.
					// plot 'j_faces.txt' w lp
					// CHECKED: It is the same DOF structure as in PoissonProblem_stationary.
					/*if (j_face_vector_global_dofs[j] != VOID_INT)
				j_faces << j_face_vector_global_dofs[j] << ' ';
			else j_faces << ' '<< ' ';*/
					for(int unsigned i = 0; i < 2; ++i) {
						j_faces << j_face_vector[j](i) << ' ';
					}
					j_faces << std::endl;
				}
			}
			j_faces.close();
		}
//		// Output "F_S,h, the set of internal faces (i.e., faces with two neighbors)" (Cut FEM Burman 14)
//		// Visualize the set of faces submitted to stabilization over p (surface var.)
		{
			std::ofstream j_faces;
			j_faces.open("j_faces_p.txt");
			for(int unsigned j = 0; j <  j_face_vector_usurface.size(); ++j) {
				if (j_face_vector_usurface[j] == VOID_POINT) {
					j_faces << std::endl;
				}
				else {
					for(int unsigned i = 0; i < 2; ++i) {
						j_faces << j_face_vector_usurface[j](i) << ' ';
					}
					j_faces << std::endl;
				}
			}
			j_faces.close();
		}
//
//		{
//			// Create a .txt file with ALL the points of the mesh
//			std::ofstream all_points;
//			all_points.open("all_points.txt");
//			for(int unsigned j = 0; j <  all_points_vector.size(); ++j) {
//				if (all_points_vector[j] == VOID_POINT) {
//					all_points << std::endl;
//				}
//				else {
//					for(int unsigned i = 0; i < 2; ++i) {
//						all_points << all_points_vector[j](i) << ' ';
//					}
//					all_points << std::endl;
//				}
//			}
//
//			all_points.close();
//		}
//		/////
//
		// Create a .txt file with all the points of each (new) cut cell face
		{
			std::ofstream new_cut_face;
			new_cut_face.open("new_cut_face.txt");
			for(int unsigned j = 0; j < new_face_vector.size(); ++j) {
				if (new_face_vector[j] == VOID_POINT) {
					new_cut_face << std::endl;
				}
				else {
					for(int unsigned i = 0; i < 2; ++i) {
						new_cut_face << new_face_vector[j](i) << ' ';
					}
					new_cut_face << std::endl;
				}
			}
			new_cut_face.close();
		}
	}


}

template <int dim>
void PoissonProblem<dim>::solve ()
{

//	The Lagrangian multiplier may be thought of as a force acting to enforce
//	the constraints. Because the zero mean value on uh is a constraint, which do not alter
//	the solution to the underlying Neumann problem, the force should vanish or, at
//	least, be very small. (The FEM, Book, Larson, Bengzon)


	SolverControl           solver_control (1000, 1e-12);
	SolverCG<>              solver (solver_control);

	if (1) {
	std::cout << "Solving full augmented Stiffness matrix "
			"with stabilization terms in U and P and constraints on P ...  \n";

	solver.solve (FM_system_matrix_block_w_constraint, /*solution_block0*/
			solution_new_w_constraint , system_rhs_block01_w_constraint,
			PreconditionIdentity());

	// Separate the solutions in U and P to output later. Eliminate the "mi" Lagrange Multiplier
	for(unsigned int i = 0; i < n_dofs_surface; ++i) {
		solution_new.block(0)(i) = solution_new_w_constraint(i);
		solution_block0(i) = solution_new_w_constraint(i);
	}

	for(unsigned int i = 0; i < n_dofs_inside; ++i) {
		solution_new.block(1)(i) = solution_new_w_constraint(i+n_dofs_surface);
	}

	std::cout << "LAGRANGE MULTIPLIER: "
			<< solution_new_w_constraint(dof_handler_new.n_dofs()) << "\n";
	}

	//	 SOLVE UBULK NO J NO CONSTRAINT
	if (0) {
		std::cout << "Solving UBULK...  NO J NO CONSTRAINT\n";
		// Here j_matrix was NOT summed into system_matrix.
		solver.solve (system_matrix_new.block(1,1), solution_new.block(1), system_rhs_new.block(1),
				PreconditionIdentity());
		std::cout << "   " << solver_control.last_step()
		  	            								<< " CG iterations needed to obtain convergence."
		  	            								<< std::endl;

		// Save solution_new_block1_old (without the constraint applied)
		{
			std::ofstream solution_new_block1_old;
			solution_new_block1_old.open("solution_new_block1_no_j_no_constraint.txt");
			for(int unsigned i = 0; i < solution_new.block(1).size(); ++i) {
				solution_new_block1_old << solution_new.block(1)(i);
				solution_new_block1_old << std::endl;
			}
			solution_new_block1_old.close();
		}
		// END SOLVE NO J NO CONSTRAINT
	}

	// SOLVE UBULK YES J (naturally without CONSTRAINT)
	if (0) {

		// Here j_matrix was summed into system_matrix.
		std::cout << "Solving UBULK...  yes_j  \n";
		solver.solve (FM_system_matrix_block11_yes_j,
				solution_new.block(1), system_rhs_new.block(1), PreconditionIdentity());

		std::cout << "   " << solver_control.last_step()
		  	  	            								<< " CG iterations needed to obtain convergence."
		  	  	            								<< std::endl;
		//
		{
			std::ofstream solution_new_block1_old;
			solution_new_block1_old.open("solution_new_block1_yes_j_no_constraint.txt");
			for(int unsigned i = 0; i < solution_new.block(1).size(); ++i) {
				solution_new_block1_old << solution_new.block(1)(i);
				solution_new_block1_old << std::endl;
			}
			solution_new_block1_old.close();
		}
	}

	// Solve USURFACE YES J NO Constraint (block 0,0)
	// Remember that the coupling term is added in the formulation;
	// If you want to really only solve the ubulk variable you have to eliminate the coupling term,
	// because it has components that are (u_b,v_b).
	if (0)
	{
//		std::cout << "Solving UBULK...  \n";
		std::cout << "Solving USURFACE...  yes_j_no_constraint\n";

		solver.solve (
//				system_matrix_new.block(0,0), // This one, without J
				FM_system_matrix_block00_w_j,
				solution_new.block(0), system_rhs_new.block(0),
				PreconditionIdentity());
		std::cout << "   " << solver_control.last_step()
	  	            								<< " CG iterations needed to obtain convergence."
	  	            								<< std::endl;
	}

	  // SOLVE USURFACE YES J YES CONSTRAINT
	if(0) {

		std::cout << "Solving USURFACE...  yes_j_yes_constraint \n";
		solver.solve (FM_system_matrix_block00_yes_j_yes_constraint, solution_block0_w_constraint
				, system_rhs_block0_w_constraint,
				PreconditionIdentity());

		std::cout << "   " << solver_control.last_step()
	  	  	            						<< " CG iterations needed to obtain convergence."
	  	  	            						<< std::endl;

		// Output Lagrange multiplier (last value of solution_w/constraint)
		std::cout << "LAGRANGE MULTIPLIER: "
				<< solution_block0_w_constraint(n_dofs_surface) << "\n";

		for(unsigned int i = 0; i < n_dofs_surface; ++i) {
			solution_block0(i)    = solution_block0_w_constraint(i);
			solution_new.block(0)(i) = solution_block0_w_constraint(i);
		}

		// Save solution_new_block1_new (WITH the constraint applied)
		{
			std::ofstream solution_new_block1_new;
			solution_new_block1_new.open("solution_new_block1_yes_j_yes_constraint.txt");
			for(int unsigned i = 0; i < solution_new.block(0).size(); ++i) {
				solution_new_block1_new << solution_new.block(0)(i);
				solution_new_block1_new << std::endl;
			}
			solution_new_block1_new.close();
		}
	}

	// SOLVE USURFACE NO J YES CONSTRAINT
	if(0) {
		// Here j_matrix was NOT summed into system_matrix.
		std::cout << "Solving USURFACE...  no_j_yes_constraint \n";
		solver.solve (FM_system_matrix_block00_no_j_yes_constraint, solution_block0_w_constraint
				, system_rhs_block0_w_constraint,
				PreconditionIdentity());

		std::cout << "   " << solver_control.last_step()
	  	  	            						<< " CG iterations needed to obtain convergence."
	  	  	            						<< std::endl;


		for(unsigned int i = 0; i < dof_handler_new.n_dofs()/2; ++i) {
			solution_new.block(1)(i) = solution_block0_w_constraint(i);
		}

		{
			std::ofstream solution_new_block1_old;
			solution_new_block1_old.open("solution_new_block1_no_j_yes_constraint.txt");
			for(int unsigned i = 0; i < solution_new.block(1).size(); ++i) {
				solution_new_block1_old << solution_new.block(1)(i);
				solution_new_block1_old << std::endl;
			}
			solution_new_block1_old.close();
		}

		// END SOLVE NO J YES CONSTRAINT
	}

	for (unsigned int i=0; i<n_dofs_surface; ++i)
	{
		exact_solution.block(0)[i] = exact_solution_usurface[i];
		difference_solution_usurface[i] = exact_solution_usurface[i] - solution_new.block(0)[i];
		difference_solution.block(0)[i] = exact_solution_usurface[i] - solution_new.block(0)[i];
	}
	for (unsigned int i=0; i<n_dofs_inside; ++i)
	{
		exact_solution.block(1)[i] 		= exact_solution_ubulk[i];
		difference_solution_ubulk[i] 	= exact_solution_ubulk[i] - solution_new.block(1)[i];
		difference_solution.block(1)[i] = exact_solution_ubulk[i] - solution_new.block(1)[i];
	}
}

template <int dim>
void PoissonProblem<dim>::output_results () const
{
	// Output uncoupled solution for usurface and ubulk
	// For coupled solution, it is the same procedure...
	{
		std::vector<std::string> solution_names;
		solution_names.push_back ("USURFACE");
		solution_names.push_back ("UBULK");

		std::vector<DataComponentInterpretation::DataComponentInterpretation>
		data_component_interpretation
		(2, DataComponentInterpretation::component_is_scalar);

		DataOut<dim,hp::DoFHandler<dim> > data_out_surface;
		data_out_surface.attach_dof_handler (dof_handler_new);
		data_out_surface.add_data_vector (solution_new, solution_names,
				DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
				data_component_interpretation);

		//    	data_out_surface.add_data_vector (solution_block1, "USURFACE");
		data_out_surface.build_patches ();

		std::string filename_new = "solution_new-";
		filename_new += ('0' + new_n_cycles);
		filename_new += ".vtk";
		std::ofstream output_new(filename_new.c_str());
		data_out_surface.write_vtk(output_new);
	}

	// Output uncoupled exact solution
    bool coupled = false;
    if ( ! coupled ) {
    {
        std::vector<std::string> solution_names;
        solution_names.push_back ("USURFACE_exact");
        solution_names.push_back ("UBULK_exact");

    	std::vector<DataComponentInterpretation::DataComponentInterpretation>
    	data_component_interpretation
    	(2, DataComponentInterpretation::component_is_scalar);

    	DataOut<dim,hp::DoFHandler<dim> > data_out_surface;
    	data_out_surface.attach_dof_handler (dof_handler_new);
    	data_out_surface.add_data_vector (exact_solution, solution_names,
    			DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
    			data_component_interpretation);

    	data_out_surface.build_patches ();
    	std::string filename_new = "exact_solution-";
    	filename_new += ('0' + new_n_cycles);
    	filename_new += ".vtk";
    	std::ofstream output_new(filename_new.c_str());
    	data_out_surface.write_vtk(output_new);
	}
    // Output uncoupled difference solution.
    {
    	std::vector<std::string> solution_names;
    	solution_names.push_back ("USURFACE_difference");
    	solution_names.push_back ("UBULK_difference");

    	std::vector<DataComponentInterpretation::DataComponentInterpretation>
    	data_component_interpretation
    	(2, DataComponentInterpretation::component_is_scalar);

    	DataOut<dim,hp::DoFHandler<dim> > data_out_surface;
    	data_out_surface.attach_dof_handler (dof_handler_new);
    	data_out_surface.add_data_vector (difference_solution, solution_names,
    			DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
    			data_component_interpretation);

    	//    	data_out_surface.add_data_vector (solution_block1, "USURFACE");
    	data_out_surface.build_patches ();

    	std::string filename_new = "difference_solution-";
    	filename_new += ('0' + cycle);
    	filename_new += ".vtk";
    	std::ofstream output_new(filename_new.c_str());
    	data_out_surface.write_vtk(output_new);
    }
    }


}

template <int dim>
void PoissonProblem<dim>::process_solution_ubulk ()
{

	  const ComponentSelectFunction<dim>	usurface_mask (0, 2); // (component, # of components)
	  const ComponentSelectFunction<dim>	ubulk_mask    (1, 2);

	QGauss<dim>  quadrature_formula(3);

    hp::QCollection<dim>  q_collection;
    q_collection.push_back (quadrature_formula);
    q_collection.push_back (quadrature_formula);

	WeightSolutionUbulk<dim> weight_values;
	// WeightSolutionUbulk takes care of eliminating the usurface vector from the computation.

	const unsigned int n_active_cells = triangulation_new.n_active_cells();
//	const unsigned int n_dofs 		  = dof_handler_new.n_dofs();

	Vector<double> weight_integrate_difference(n_active_cells); // Vector initiliazed, all elements equal to zero.
	ALLERRORS_ubulk[cycle][0] = cycle;
	ALLERRORS_ubulk[cycle][1] = n_active_cells;
	ALLERRORS_ubulk[cycle][2] = n_dofs_inside;
	Vector<float> difference_per_cell (triangulation_new.n_active_cells());
	VectorTools::integrate_difference(dof_handler_new,
			solution_new,
			ExactSolution<dim>(),
			difference_per_cell,
			q_collection,
			VectorTools::L2_norm
			,&weight_values
//			,&ubulk_mask
	);

	const double L2_error = difference_per_cell.l2_norm();
	ALLERRORS_ubulk[cycle][4] = L2_error;
	VectorTools::integrate_difference (dof_handler_new,
			solution_new,
			ExactSolution<dim>(),
			difference_per_cell,
			q_collection,
			VectorTools::H1_norm
			,&weight_values
	);
	const double H1_error = difference_per_cell.l2_norm();
	//    H1_norm :The square of this norm is the square of the L2_norm plus the square of the H1_seminorm.

	ALLERRORS_ubulk[cycle][5] = H1_error;
	const QTrapez<1>     q_trapez;
	const QIterated<dim> q_iterated (q_trapez, 5);

    hp::QCollection<dim>  q_iterated_collection;
    q_iterated_collection.push_back (q_iterated);
    q_iterated_collection.push_back (q_iterated);
	VectorTools::integrate_difference (	dof_handler_new,
			solution_new,
			ExactSolution<dim>(),
			difference_per_cell,
			q_iterated_collection,
			VectorTools::Linfty_norm
			,&weight_values
	);
	const double Linfty_error = difference_per_cell.linfty_norm();
	ALLERRORS_ubulk[cycle][6] = Linfty_error;
}

template <int dim>
void PoissonProblem<dim>::interpolate_solution_usurface (){

	// levelset_face_map is a map containing the angle of the point where the levelset
	// function cuts the cell and the coordinates of the point itself.
	// As the map is automatically ordered with ascending order by the key members, the levelset points
	// get ordered in a clockwise manner (it doesn't really matter if it is clock- or anticlockwise,
	// they just need to be connected in order to create the mesh)
	// once the levelset points are ordered, one needs only to create a vector with these points to
	// use as input for create_triangulation.
	std::map <double, Point<2> >::iterator it;
	std::vector <Point<2>> levelset_face_vertices_tangent;
    for(it = levelset_face_map.begin(); it != levelset_face_map.end(); ++it )
    	levelset_face_vertices_tangent.push_back( it->second );
    // CellData struct used to input the vertex numbering. Note that the nodes are numbered such as:
    // (0 1),(1 2),(2 3)... (n-2 0). The last vertex needs to match the first.
    // The numbering itself doesn't need to have a logical structure, but the coincident nodes of
    // different elements must have the same number
	std::vector<CellData<1> > cells_data (levelset_face_map.size(), CellData<1>());
	for (unsigned int i=0; i< levelset_face_map.size(); ++i)
		for (unsigned int j=0; j< 2; ++j)
			cells_data[i].vertices[j] = i+j;

	cells_data[levelset_face_map.size()-1].vertices[1] = 0;

	// Create triangulation with codimension 1 and space dimension 2.
	Triangulation<1,2> levelset_triangulation;
	levelset_triangulation.create_triangulation
		(levelset_face_vertices_tangent, cells_data, SubCellData());

	DoFHandler<1,2>      dof_handler_inside(levelset_triangulation);
	dof_handler_inside.distribute_dofs (fe_dummy);

	// Vector for the interpolated solution on the boundary.
	Vector<double> solution_inside(dof_handler_inside.n_dofs());
	QGauss<dim-1> quadrature_formula(2);

	// Interpolate the values from solution_new on the points given by the levelset triangulation.
	std::vector<unsigned int> local_dof_indices(fe_dummy.dofs_per_cell); // dofs_per_cell = 2
	Vector< double > interpolated_values(fe_dummy.dofs_per_cell);
	Point<dim> point_for_interpolation;
	for (typename DoFHandler<dim-1,dim>::active_cell_iterator
			cell = dof_handler_inside.begin_active();
			cell != dof_handler_inside.end(); ++cell) {
//		fe_dummy_values.reinit(cell);
		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i<fe_dummy.dofs_per_cell; ++i) {
			point_for_interpolation = cell->vertex(i);
			VectorTools::point_value (dof_handler_new, solution_new, point_for_interpolation,
					interpolated_values );
			// Remember that dof_handler_new is vector valued: [0] for usurface, [1] for ubulk.
			solution_inside(local_dof_indices[i]) = interpolated_values[0];
		}
	}

	MappingQ<1, 2>   mapping_inside(1);

	{
		DataOut<1,DoFHandler<1,2> > data_out;
		data_out.attach_dof_handler (dof_handler_inside);
		data_out.add_data_vector (solution_inside, "solution_inside",
				DataOut<1,DoFHandler<1,2> >::type_dof_data);
		data_out.build_patches (mapping_inside,	mapping_inside.get_degree());

		std::string filename ("usurface_interpolated-");
		filename += static_cast<char>('0'+new_n_cycles);
		filename += ".vtk";
		std::ofstream output (filename.c_str());
		data_out.write_vtk (output);
	}

	// Create exact solution...
	Vector<double> exact_solution_inside(dof_handler_inside.n_dofs());
	VectorTools::interpolate(dof_handler_inside, ExactSolutionUsurface<dim>(),
			exact_solution_inside);
	// Create difference solution inside...
	Vector<double> difference_solution_inside(dof_handler_inside.n_dofs());
	difference_solution_inside += solution_inside;
	difference_solution_inside -= exact_solution_inside;
	// Output difference solution
	{
		DataOut<1,DoFHandler<1,2> > data_out;
		data_out.attach_dof_handler (dof_handler_inside);
		data_out.add_data_vector (difference_solution_inside, "difference_solution_inside",
				DataOut<1,DoFHandler<1,2> >::type_dof_data);
		data_out.build_patches (mapping_inside,	mapping_inside.get_degree());

		std::string filename ("difference_solution_inside-");
		filename += static_cast<char>('0'+new_n_cycles);
		filename += ".vtk";
		std::ofstream output (filename.c_str());
		data_out.write_vtk (output);
	}

	QGauss<dim-1> quadrature_formula_error(2);

	Vector<double> difference_per_cell(levelset_triangulation.n_active_cells());

	// Integrate_difference function asks for an ExactSolution in <spacedim> dimensions, which is
	// dim = 2 here
	VectorTools::integrate_difference(
			dof_handler_inside,  		  // dim = 1, spacedim = 2
			solution_inside,
			ExactSolutionUsurface<dim>(), // spacedim = 2
			difference_per_cell,
			quadrature_formula, 		 // dim = 1
			VectorTools::L2_norm );

	const double L2_norm = difference_per_cell.l2_norm();
	std::cout << "L2_norm: " << L2_norm << "\n";

	VectorTools::integrate_difference(
				dof_handler_inside,  		  // dim = 1, spacedim = 2
				solution_inside,
				ExactSolutionUsurface<dim>(), // spacedim = 2
				difference_per_cell,
				quadrature_formula, 		 // dim = 1
				VectorTools::H1_norm );

	const double H1_error = difference_per_cell.l2_norm();
	std::cout << "H1_error: " << H1_error << "\n";

	const QTrapez<1>     q_trapez;
	const QIterated<dim> q_iterated (q_trapez, 5);

	ALLERRORS_usurface[cycle][0] = new_n_cycles;
	ALLERRORS_usurface[cycle][1] = triangulation_new.n_active_cells();
	ALLERRORS_usurface[cycle][2] = n_dofs_surface;

	ALLERRORS_usurface[cycle][4] = L2_norm;
	ALLERRORS_usurface[cycle][5] = H1_error;
}

template <int dim>
void PoissonProblem<dim>::run ()
{
	std::cout << "===================== \n ";
	std::cout << "START NEW CYCLE = " << new_n_cycles << "\n";
	std::cout << "===================== \n ";
	cycle = 0;
	n_cycles = 1;

	ALLERRORS_ubulk.reinit(n_cycles,8);
	ALLERRORS_usurface.reinit(n_cycles,8);

	while (cycle < n_cycles)
	{
		make_grid (); // This will make the first grid (cycle =0) and then just refine. Updated to cycle
		initialize_levelset();
		output_results_levelset();
		get_new_triangulation ();

		setup_system_new();
		initialize_levelset_new();
		assemble_system_newMesh();
		solve ();
		output_results ();
		process_solution_ubulk();
		interpolate_solution_usurface();
		cycle++;
	}
	// Write all error info. results in a .txt file (after all cycles are completed)
	// File used for error visualization: ErrorEvaluationGraphs.ods
	//	Order of output:
	//	Cycle|n_active_cells|n_dofs|cell_diameter|L2 Error|H1 Error|Linf Error|L1 Cond.Number
    bool coupled = false;
    if (! coupled ) {
    	{
    		std::ofstream write_output("ErrorEvaluation_usurface_appending.txt", std::ios::app);
    		assert(write_output.is_open());

    		for(int unsigned i = 0; i < n_cycles; ++i) {
    			for(int unsigned j = 0; j < ALLERRORS_usurface.size()[1]; ++j) {
    				write_output << ALLERRORS_usurface[i][j] << ',';
    			}
    			write_output << std::endl;
    		}
    		write_output.close();
    	}

    	{
    		std::ofstream write_output("ErrorEvaluation_ubulk_appending.txt", std::ios::app);
    		assert(write_output.is_open());

    		for(int unsigned i = 0; i < n_cycles; ++i) {
    			for(int unsigned j = 0; j < ALLERRORS_ubulk.size()[1]; ++j) {
    				write_output << ALLERRORS_ubulk[i][j] << ',';
    			}
    			write_output << std::endl;
    		}
    		write_output.close();
    	}
    }
}
} // End namespace

int main ()
{
	using namespace dealii;
	using namespace cut_cell_method;
	const unsigned int dim = 2;
	int _n_cycles = 5;
	system("exec rm ErrorEvaluation_usurface_appending.txt");
	system("exec rm ErrorEvaluation_ubulk_appending.txt");
	for(int i = 0; i < _n_cycles; ++i) {
		PoissonProblem<dim> Obj_PoissonProblem(i);
		Obj_PoissonProblem.run ();
	}

	return 0;
}
