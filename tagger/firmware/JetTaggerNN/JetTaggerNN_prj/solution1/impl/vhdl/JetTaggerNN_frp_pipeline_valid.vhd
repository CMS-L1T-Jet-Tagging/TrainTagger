-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
-- Tool Version Limit: 2019.12
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- ==============================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library STD;
use STD.textio.all;

entity JetTaggerNN_frp_pipeline_valid is
generic (
    PipelineLatency : INTEGER := 4;
    PipelineII      : INTEGER := 1;
    ExitLatency    : INTEGER := -1;
    CeilLog2Stages      : INTEGER := 2);
port (
    ap_clk      	: IN STD_LOGIC;
    ap_rst      	: IN STD_LOGIC;
    valid_in    	: IN STD_LOGIC;
    exitcond    	: IN STD_LOGIC;

    valid_out    	: OUT STD_LOGIC_VECTOR( PipelineLatency-1 downto 0);
    num_valid_datasets    : OUT STD_LOGIC_VECTOR( CeilLog2Stages downto 0));
end;

architecture behav of JetTaggerNN_frp_pipeline_valid is
    --	(* keep_hierarchy = "yes" *)
    attribute KEEP_HIERARCHY : STRING;
    attribute KEEP_HIERARCHY of behav :architecture is "TRUE";
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';

   
   --	localparam NUM_STAGES = (PipelineLatency+PipelineII-1) / PipelineII; // Ceil(Latency/II)
   constant NUM_STAGES : INTEGER := (PipelineLatency+PipelineII-1) / PipelineII; --	// Ceil(Latency/II)
 
   
  --	function isDefined;
  --	   input val;
  --	   begin
		-- return (val === 1'b0 || val === 1'b1);
  --	   end
  --	endfunction
	function isDefined(val : STD_LOGIC) return BOOLEAN is
		variable defi : BOOLEAN;
	begin
			defi := FALSE;
			--	return (val === 1'b0 || val === 1'b1);
			if((val = ap_const_logic_1) OR (val = ap_const_logic_0)) then
				defi := TRUE; 
			end if;
			return defi;
	end function isDefined;
	
   signal nvd_reg : STD_LOGIC_VECTOR(CeilLog2Stages +1 downto 0) := (others=>'0');
   shared variable fail : STD_LOGIC := '0';
  
   signal v2 : STD_LOGIC_VECTOR( PipelineLatency-2 downto 0) := (others=>'0');
 
   signal valid_out_tmp : STD_LOGIC_VECTOR( PipelineLatency-1 downto 0);
   
   signal valid_disable : STD_LOGIC;
   
   signal stg : INTEGER := 0;
   signal stgc : INTEGER := 0;
   
begin
	--	VHDL functional codes begin here.	$huic


   --	assign num_valid_datasets=nvd_reg;
   num_valid_datasets <= nvd_reg(CeilLog2Stages downto 0);


   valid_out <= valid_out_tmp;
   --	///////// pipeline data valid (flushable impl) /////////
   --	wire [PipelineLatency-2:0] v1;
   --	genvar k;
      
  
	v1_v2_gen : for i in 0 to PipelineLatency-2 generate
		process (ap_clk)
			variable v1 : STD_LOGIC_VECTOR( PipelineLatency-2 downto 0);
		begin
			if ap_clk'event and ap_clk = '1' then
				if (ap_rst = '1') then
					v2(i) <= '0';
				else
					if (i = 0) then
						v1(i) := valid_in;
					else
						v1(i) := v2(i-1);
					end if;
					
					if (i = ExitLatency) then
						--	v2(i) <= STD_LOGIC((not exitcond) and (v1(i) = ap_const_logic_1));
						if ((exitcond = ap_const_logic_0) and (v1(i) = ap_const_logic_1)) then
							v2(i) <= ap_const_logic_1;
						else
							v2(i) <= ap_const_logic_0;
						end if;
						
					else
						v2(i) <= v1(i);
					end if;
					
				------	--	// synthesis translate_off
				------	if (i = 0) then
				------		if ( isDefined(valid_in) ) then
				------		else
				------			--	ASSERT FALSE REPORT "valid_in is " & valid_in & "." SEVERITY ERROR;
				------			fail := '1';
				------		end if;
				------	end if;
					------  // synthesis translate_on
				end if;
			end if;
		end process;
	end generate v1_v2_gen;

   
   process (v2,valid_in)
   begin
		valid_out_tmp <= v2 & valid_in;

	------	--	// synthesis translate_off
	------	if (now > 0ps) then
	------		for j in 1 to PipelineLatency - 1 loop
	------			if ( isDefined(valid_out_tmp(j)) ) then
	------			else
	------				--	ASSERT FALSE REPORT "valid_out_tmp[" & integer'image(j) & "] is " & valid_out_tmp(j) & " at time " & time'image(now) SEVERITY ERROR;
	------				fail := '1';
	------				exit;
	------			end if;
	------			
	------			if ( (valid_out_tmp(j) = ap_const_logic_0) or (conv_integer(nvd_reg) <= NUM_STAGES) ) then
	------			else
	------				--	ASSERT FALSE REPORT "nvd_reg overflow. valid_out_tmp[" & integer'image(j) & "]=" &  valid_out_tmp(j) & "nvd=" & nvd_reg & " nstg=" & NUM_STAGES & "(at time " & time'image(now) & ")" SEVERITY ERROR;
	------				fail := '1';
	------				exit;
	------			end if;
	------	
	------		end loop;
	------	end if;
		-----  // synthesis translate_on
   
   end process;

   --	wire   valid_disable;
   --	assign valid_disable = (ExitLatency == -1) ? 1'b0 : valid_out_tmp[ExitLatency];
   valid_disable_gen1: if (ExitLatency = -1) generate
   begin
       valid_disable <= ap_const_logic_0;
   end generate;

   valid_disable_gen2: if (ExitLatency >= 0) generate
   begin
       valid_disable <= valid_out_tmp(ExitLatency) and exitcond;
   end generate;
   
   --	// Count the num valid_out_tmp datasets in the pipe stages other than the
   --	// first. We do NOT count valid_out_tmp[0], otherwise it will be a
   --	// combination loop: nvd -> pfifo.ap_ready -> valid_in -> valid_out_tmp[0] -> nvd
   --	reg signed [2:0] incr;
  
    nvd_reg_pro : process (ap_clk)
		variable incr : STD_LOGIC_VECTOR(2 downto 0);
		variable case_tmp : STD_LOGIC_VECTOR(2 downto 0);
	begin
			case_tmp := valid_out_tmp(0) & valid_out_tmp(PipelineLatency-1) & valid_disable;
			
			if ap_clk'event and ap_clk = '1' then
				if (ap_rst = ap_const_logic_1) then
					nvd_reg <= (others => '0');
				else
					case case_tmp is
						when "100" => incr := STD_LOGIC_VECTOR(TO_SIGNED( 1, incr'length ));
						when "111" => incr := STD_LOGIC_VECTOR(TO_SIGNED(-1, incr'length ));
						when "001" => incr := STD_LOGIC_VECTOR(TO_SIGNED(-1, incr'length ));
						when "010" => incr := STD_LOGIC_VECTOR(TO_SIGNED(-1, incr'length ));
						when "011" => incr := STD_LOGIC_VECTOR(TO_SIGNED(-2, incr'length ));
						when others => incr :=STD_LOGIC_VECTOR(TO_SIGNED( 0, incr'length ));
					end case;
					
					----	// synthesis translate_off
					----	assert( 0 <= ($signed(nvd_reg) + incr) <= NUM_STAGES ) else begin
					----		$error("nvd_reg out of range. incr=%d nvd=%0d nstg=%0d (at time %0t)",
					----				incr, nvd_reg, NUM_STAGES, $time);
					----		fail=1;
					----	end
					----	// synthesis translate_on
			
				--	if ( (0 <= (conv_integer(nvd_reg) + conv_integer(incr))) and ((conv_integer(nvd_reg) + conv_integer(incr)) <= NUM_STAGES) ) then
				--	else
				--		--	ASSERT FALSE REPORT "nvd_reg out of range. incr=" & incr & "nvd=" & nvd_reg & "nstg=" & NUM_STAGES & "(at time " & time'image(now) & ")." SEVERITY ERROR;
				--		fail := '1';
				--		--	exit;
				--	end if;
			
			
					--	nvd_reg <= conv_STD_LOGIC_VECTOR( (conv_integer(nvd_reg) + conv_integer(incr)), nvd_reg'length);
					nvd_reg <= STD_LOGIC_VECTOR( SIGNED(nvd_reg) + SIGNED(incr));
				end if;
			end if;
	end process;


	-----	// synthesis translate_off
------   process (ap_clk)
------   begin
------		if ap_clk'event and ap_clk = '1' then
------			for j in 1 to PipelineLatency - 1 loop
------				stgc <= stgc + conv_integer(valid_out_tmp(j));
------				if ( stg = PipelineII-1 ) then
------					if ( stgc <= 1 ) then
------					else
------						--	ASSERT FALSE REPORT "valid_out_tmp=" & valid_out_tmp & '.' & stgc & " bits set in one pipe-stage ending at bit-" & j & " (at time" & time'image(now) & ")."	SEVERITY ERROR;
------						fail := '1';
------						exit;
------					end if;
------					stgc <= 0;
------					stg  <= 0;
------				else
------					stg <= stg + 1;
------				end if;
------			
------			end loop;	
------		end if;
------   end process;
   
   
------   process
------   begin
------      if (fail = ap_const_logic_1) then
------		ASSERT FALSE REPORT "<<<<<<<< Abort! Assertion failure at time " & time'image(now) & "  >>>>>>>>" SEVERITY FAILURE;
------		wait until rising_edge(ap_clk);
------		wait until rising_edge(ap_clk);
------      end if;
------   end process;

--	endmodule // pipeline_valid

end behav;
    
