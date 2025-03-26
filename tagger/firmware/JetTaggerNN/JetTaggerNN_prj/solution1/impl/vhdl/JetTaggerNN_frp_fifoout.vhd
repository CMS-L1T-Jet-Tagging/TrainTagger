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

entity JetTaggerNN_frp_fifoout is
generic (
    BlockingType : INTEGER := 1;  -- 'frp_fifoout' type parameter: '1' for blocking port(AXIS/FIFO/AP_HS)
                                  --                               '0' for non-blocking port(Memory/AP_VLD...)
    PipeLatency : INTEGER := 4;
    PipelineII  : INTEGER := 1;
    DataWidth   : INTEGER := 32;
    CeilLog2Stages  : INTEGER := 2;
    CeilLog2FDepth  : INTEGER := 2;
    NumWrites   : INTEGER := 1; -- per transaction (in the last pipe stage)
    PfAllDoneEnable : INTEGER := 2
    );
port (
    ap_clk			: IN STD_LOGIC;
    ap_rst			: IN STD_LOGIC;
    data_in 		: IN STD_LOGIC_VECTOR (DataWidth-1 downto 0);
    data_in_vld 	: IN STD_LOGIC;
    data_in_last 	: IN STD_LOGIC;
    data_out		: OUT STD_LOGIC_VECTOR (DataWidth-1 DOWNTO 0);
    data_out_vld 	: OUT STD_LOGIC;
    
    ap_start 		: IN STD_LOGIC;
    data_out_read 	: IN STD_LOGIC;
    
    valid 			: IN STD_LOGIC_VECTOR (PipeLatency-1 downto 0);
    num_valid_datasets : IN STD_LOGIC_VECTOR (CeilLog2Stages downto 0);
    pf_ready 		: OUT STD_LOGIC;
    pf_done 		: OUT STD_LOGIC;
    pf_all_done     : IN STD_LOGIC;
    pf_continue     : IN STD_LOGIC;
    empty 			: OUT STD_LOGIC);
end;

architecture behav of JetTaggerNN_frp_fifoout is

    attribute KEEP_HIERARCHY : STRING;
    attribute KEEP_HIERARCHY of behav : architecture is "TRUE";

    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant NUM_STAGES : INTEGER := (PipeLatency+PipelineII-1) / PipelineII; -- Ceil(Latency/II)

	function isDefined(val : STD_LOGIC) return BOOLEAN is
		variable defi : BOOLEAN;
	begin
			defi := FALSE;
			if((val = ap_const_logic_1) OR (val = ap_const_logic_0)) then
				defi := TRUE; 
			end if;
			return defi;
	end function isDefined;

   --	///////// custom fifo implementation /////////
   --	localparam FDEPTH = NUM_STAGES * NumWrites;
   
   --  custom fifo implementation
   constant FDEPTH : INTEGER := NUM_STAGES * NumWrites;
   type FIFO_type is array (FDEPTH-1 downto 0) of STD_LOGIC_VECTOR(DataWidth-1+2 DOWNTO 0);
   signal fifo : FIFO_type;
   signal write_enable : STD_LOGIC;
   signal read_enable : STD_LOGIC;
   signal fifo_empty : STD_LOGIC := '1';
   signal fifo_full : STD_LOGIC := '0';
   signal sel_pipe : STD_LOGIC;

   signal fifo_rdPtr : STD_LOGIC_VECTOR(CeilLog2FDepth downto 0);
   signal fifo_avail : STD_LOGIC_VECTOR(CeilLog2FDepth downto 0);
   signal ready_chk : STD_LOGIC_VECTOR(CeilLog2FDepth+1 downto 0);
   
   shared variable fail : STD_LOGIC;
   
   signal fifo_out : STD_LOGIC_VECTOR(DataWidth-1+2 DOWNTO 0);
   
   signal internal_done : STD_LOGIC;

   signal stalled : STD_LOGIC;
   signal must_stall : STD_LOGIC;

   signal data_cannot_vld : STD_LOGIC;
   signal data_done       : STD_LOGIC;
   signal blocking_type   : STD_LOGIC;
   signal data_out_last : STD_LOGIC;
   signal data_out_last_reg : STD_LOGIC;

   signal pf_ready_s : STD_LOGIC;
   signal pf_done_s : STD_LOGIC;
   signal data_out_vld_s : STD_LOGIC;
   signal pf_continue_done : STD_LOGIC;

begin	
   
   fifo_gen : process (ap_clk)
   begin
		if (ap_clk'event and ap_clk = '1') then
			if (write_enable = '1') then
				for i in 0 to FDEPTH-2 loop
					fifo(i+1) <= fifo(i);
				end loop;
				fifo(0) <= data_in_last & data_in_vld & data_in;
			end if;
		end if;
   end process;
 
   
   fifo_out_gen : process (fifo_rdPtr, fifo)
		variable idx : integer;
   begin
		idx := TO_INTEGER(RESIZE(UNSIGNED(fifo_rdPtr),CeilLog2FDepth));
		if (idx < 0 or idx >= FDEPTH) then
			fifo_out <= (others => 'X');
		else
			fifo_out <= fifo(idx);
		end if;
   end process fifo_out_gen;
   
    
   fifo_ptr : process (ap_clk)
   begin
		if ap_clk'event and ap_clk = '1' then
			if ap_rst = '1' then
				fifo_rdPtr <= STD_LOGIC_VECTOR(TO_SIGNED(-1, fifo_rdPtr'length));
				fifo_empty <= ap_const_logic_1;
				fifo_full <= ap_const_logic_0;
			else
			-----	--	// synthesis translate_off
			-----	if (now > 0ps) then
			-----		if ( NOT ((write_enable = ap_const_logic_1) AND (fifo_full = ap_const_logic_1)) ) then
			-----		else
			-----			ASSERT FALSE REPORT "fifo write overflow at time " & time'image(now) SEVERITY ERROR;
			-----			fail := '1';
			-----		end if;
			-----	end if;
			---	--  // synthesis translate_on
				if ((write_enable = ap_const_logic_1) and (read_enable = ap_const_logic_0) and (fifo_full = ap_const_logic_0)) then
					fifo_rdPtr <= STD_LOGIC_VECTOR(signed(fifo_rdPtr) + To_signed(1,CeilLog2FDepth));
					fifo_empty <= ap_const_logic_0;
					if (TO_INTEGER( SIGNED(fifo_rdPtr) ) = (FDEPTH - 2)) then
						fifo_full <= ap_const_logic_1;
					else
						fifo_full <= ap_const_logic_0;
					end if;
				elsif ((read_enable = ap_const_logic_1) and (write_enable = ap_const_logic_0) and (fifo_empty = ap_const_logic_0)) then
					fifo_rdPtr <= STD_LOGIC_VECTOR(signed(fifo_rdPtr) - To_signed(1,CeilLog2FDepth));
					if (TO_INTEGER( SIGNED(fifo_rdPtr) ) = 0) then
						fifo_empty <= ap_const_logic_1;
					else
						fifo_empty <= ap_const_logic_0;
					end if;
					fifo_full <= ap_const_logic_0;
				end if;
			end if;
		end if;
   end process;
   
   --	///////// control logic /////////
   
    stall_wire_gen : if(PfAllDoneEnable=2) generate 
        stall_wire_proc : process (must_stall)
        begin
             stalled <= must_stall;
        end process;
    end generate stall_wire_gen;

    stall_reg_gen : if(PfAllDoneEnable=0 or PfAllDoneEnable=1) generate 
        stall_reg_proc : process (ap_clk)
        begin
         	if ap_clk'event and ap_clk='1' then
         		if ap_rst = '1' then
         			stalled <= ap_const_logic_0;
         		else
         			stalled <= must_stall;
         		end if;
         	end if;
        end process;
    end generate stall_reg_gen;
   
   --	///////// control logic /////////
    readen_control_for_blocking : if (BlockingType = 1) generate
        PROC_READ_ENABLE : process (data_out_read, data_out_vld_s, data_out_last, fifo_empty, stalled)
            variable read_enable_casex : STD_LOGIC_VECTOR(4 downto 0);
        begin
            read_enable_casex := data_out_read & data_out_vld_s & data_out_last & fifo_empty & stalled;
            case read_enable_casex is 
                when "10000" | "10100" | "11000" | "11100" =>  -- 4'b 1_X_X_0 :when consumer reads and fifo is not empty.
                    read_enable <= '1';
                when "00100" =>  -- 4'b X_0_1_0: when no data in last and fifo is not empty. "1010" is covered.
                    read_enable <= '1';
                when others =>  
                    read_enable <= '0';
            end case;    
        end process;
    end generate readen_control_for_blocking;

    readen_control_for_non_blocking : if (BlockingType = 0) generate
        PROC_READ_ENABLE : process (fifo_empty, must_stall)
            variable read_enable_casex : STD_LOGIC_VECTOR(1 downto 0);
        begin
            read_enable_casex := fifo_empty & must_stall;
            case read_enable_casex is 
                when "00" =>
                    read_enable <= '1';
                when others =>  
                    read_enable <= '0';
            end case;    
        end process;
    end generate readen_control_for_non_blocking;
    
    writen_control_for_blocking : if (BlockingType = 1) generate
        process (data_out_read, data_in_vld, fifo_empty, data_in_last, stalled)
            variable case_select_for_write_en : STD_LOGIC_VECTOR(4 downto 0);
        begin
            --  write_enable logic generation. 
            case_select_for_write_en := data_out_read & data_in_vld & fifo_empty & data_in_last & stalled;
            case case_select_for_write_en is
                when "00011" | "00111" | "01011" | "01111" | "10011" | "10111" | "11011" | "11111"
                                   => write_enable <= '1';
                when "01001" | "01101" | "11001" | "11101" 
                                   => write_enable <= '1';
                                   
	    		when "00010" | "01010" | "10010" | "11010" -- last valid, fifo has data
                                   => write_enable <= '1';
	    		when "01110"                            -- last valid, but consumer stalls (data_out_read==0), data_in_valid available     
                                   => write_enable <= '1';
        
	    		when "01000" | "01100"  -- data valid, but consumer stalls (data_out_read==0); "0101","0111" are covered before.       
                                   => write_enable <= '1';
	    		when "11100" | "11110"   --  data valid, consumer ready, fifo is fifo_empty.  
                                   => write_enable <= '0';
	    		when "11000"  --  data valid, consumer ready, fifo has data. "1101" is covered before.
                                   => write_enable <= '1';
	    		when others        => write_enable <= '0';
	    	end case;
        end process;
    end generate writen_control_for_blocking;
    
    writen_control_for_non_blocking : if (BlockingType = 0 and PfAllDoneEnable /=2) generate
        process (data_in_vld, fifo_empty, data_in_last, must_stall)
            variable case_select_for_write_en : STD_LOGIC_VECTOR(3 downto 0);
        begin
            --  write_enable logic generation. 
            case_select_for_write_en := data_in_vld & fifo_empty & data_in_last & must_stall;
            case case_select_for_write_en is
	    		when "1000" | "1001" | "1010" | "1011"  -- 4'b 1_0_X_X :
                                   => write_enable <= '1';
	    		when "0010" | "0011"  -- 4'b X_0_1_X :
                                   => write_enable <= '1';
	    		when "0111" | "1111"  -- 4'b X_1_1_1 :
                                   => write_enable <= '1';
	    		when others        => write_enable <= '0';
	    	end case;
        end process;
    end generate writen_control_for_non_blocking;

    writen_control_for_non_blocking_pfalldonereg : if (BlockingType = 0 and PfAllDoneEnable =2) generate
        process (data_in_vld, fifo_empty, data_in_last, must_stall)
            variable case_select_for_write_en : STD_LOGIC_VECTOR(3 downto 0);
        begin
            --  write_enable logic generation. 
            case_select_for_write_en := data_in_vld & fifo_empty & data_in_last & must_stall;
            case case_select_for_write_en is
	    		when "1000" | "1001" | "1010" | "1011"  -- 4'b 1_0_X_X :
                                   => write_enable <= '1';
	    		when "0010" | "0011"  -- 4'b X_0_1_X :
                                   => write_enable <= '1';
	    		when "0111" | "1111" | "1101" -- 4'b X_1_1_1 :
                                   => write_enable <= '1';
	    		when others        => write_enable <= '0';
	    	end case;
        end process;
    end generate writen_control_for_non_blocking_pfalldonereg;

    block_type_sig1_gen : if(BlockingType = 1) generate
        blocking_type <= '1';
    end generate block_type_sig1_gen;

    block_type_sig0_gen : if(BlockingType = 0) generate
        blocking_type <= '0';
    end generate block_type_sig0_gen;

    pfalldone_noiidelay_reg_gen :  if (PfAllDoneEnable = 2) generate
        process (ap_clk) 
        begin
            if (ap_clk'event and ap_clk='1') then
                if ap_rst = '1' then
                    data_cannot_vld<=ap_const_logic_0;
                    data_out_last_reg<=ap_const_logic_0;
                elsif(pf_all_done = '1' and pf_continue = '1') then 
                    if(data_out_vld_s='1' and data_out_read='1' and data_out_last='1') then
                        if(blocking_type='1') then 
                            data_cannot_vld<=ap_const_logic_1;
                        else
                            data_cannot_vld<=ap_const_logic_0;
                        end if;
                    elsif(data_out_vld_s='0' and data_out_last='1') then
                        data_cannot_vld<='1';
                    else
                        data_cannot_vld<=ap_const_logic_0;
                    end if;

                    if(data_out_vld_s='1' and data_out_read='1' and data_out_last='1') then
                        if(blocking_type='1') then
                            data_out_last_reg<=data_out_last;
                        else
                            data_out_last_reg<=ap_const_logic_0;
                        end if;
                    elsif(data_out_vld_s='0' and data_out_last='1') then
                            data_out_last_reg<=data_out_last;
                    else 
                        data_out_last_reg<='0';
                    end if;
                elsif(pf_all_done='1' and pf_continue='0') then 
                    data_cannot_vld<=ap_const_logic_1;
                    data_out_last_reg<=data_out_last_reg;
                elsif(pf_all_done='0') then
                    if(data_out_last_reg='1') then
                        data_cannot_vld<=ap_const_logic_1;
                        data_out_last_reg<=data_out_last_reg;
                    elsif(data_out_vld_s='1' and data_out_read='0') then
                        data_cannot_vld<=ap_const_logic_0;
                        data_out_last_reg<=ap_const_logic_0;
                    elsif((data_out_vld_s='1' and data_out_read='1') or data_out_vld_s='0') then
                        if(data_out_last='0') then
                            data_cannot_vld<=ap_const_logic_0;
                        else 
                            data_cannot_vld<=ap_const_logic_1;
                        end if;

                        if(data_out_last='1') then
                            data_out_last_reg<=data_out_last;
                        else 
                            data_out_last_reg<=ap_const_logic_0;
                        end if;
                    end if;
                end if;

                if ap_rst = '1' then
                    data_done <= ap_const_logic_0;
                elsif(blocking_type='0') then
                    data_done <= ap_const_logic_0;
                elsif(must_stall = '1') then
                    data_done <= data_done;
                elsif( ((data_out_vld_s='1' and data_out_read='1') or (data_out_vld_s='0' and data_out_last='1')) or
                       (pf_all_done='0' or (pf_all_done='1' and pf_continue='1')) ) then 
                    data_done <= ap_const_logic_1;
                else 
                    data_done <= ap_const_logic_0;
                end if;
            end if;
        end process;
    end generate pfalldone_noiidelay_reg_gen;

    pf_continue_done_with_alldone : if (PfAllDoneEnable = 1) generate
        pf_continue_done <= pf_continue and pf_all_done;
    end generate pf_continue_done_with_alldone;
    
    pf_continue_done_without_alldone : if (PfAllDoneEnable = 0) generate
        pf_continue_done <= pf_continue;
    end generate pf_continue_done_without_alldone;

    pf_done_with_alldone_noiidelay : if (PfAllDoneEnable = 2) generate
   control : process (fifo_empty, data_out_read, data_in_vld, fifo_rdPtr, fifo_avail, fifo_out, pf_ready_s, ready_chk, data_in_last, data_in, data_out_last, data_out_last_reg, pf_continue, pf_all_done, data_done, data_cannot_vld, data_out_vld_s, must_stall, read_enable, num_valid_datasets, sel_pipe)
		variable case_select : STD_LOGIC_VECTOR(4 downto 0);
        variable case_select_for_write_en : STD_LOGIC_VECTOR(3 downto 0);
    begin 
        --	///////// internal signals /////////
        sel_pipe <= fifo_empty;
        if (sel_pipe = ap_const_logic_1) then
          	data_out_last <= data_in_last;
          	data_out_vld_s <= data_in_vld;
          	data_out <= data_in;
        else
          	data_out_last <= fifo_out(DataWidth + 1);
          	data_out_vld_s <= fifo_out(DataWidth);
          	data_out <= fifo_out(DataWidth-1 DOWNTO 0);
        end if;

        data_out_vld <= data_out_vld_s and ((not data_cannot_vld) or (pf_continue and pf_all_done and data_done));

        if(pf_continue='1' and pf_all_done='1')  then
            if(data_out_vld_s='1' and data_out_read='0') then
                must_stall<=ap_const_logic_0;
            elsif(data_out_vld_s='1' and data_out_read='1') then
                if(blocking_type='0') then
                    must_stall<=ap_const_logic_1;
                else
                    must_stall<=ap_const_logic_0;
                end if;
            else 
                must_stall<=ap_const_logic_0;
            end if;

            if(data_out_vld_s='1' and data_out_read='1') then
                if(blocking_type='1') then
                    pf_done<=data_out_last;
                else
                    pf_done<=ap_const_logic_0;
                end if;
            elsif(data_out_vld_s='0' and data_out_last='1') then
                pf_done<=data_out_last;
            else 
                pf_done<=ap_const_logic_0;
            end if;
        elsif(pf_all_done='1' and pf_continue='0')  then 
            must_stall<=ap_const_logic_1;
            pf_done<=data_out_last_reg;
        elsif(pf_all_done='0')  then
            if(data_out_last_reg='1')  then
                 must_stall<=ap_const_logic_1;
                 pf_done<=data_out_last_reg;
            elsif(data_out_vld_s='1' and data_out_read='0') then
                must_stall<=ap_const_logic_1;
                pf_done<=ap_const_logic_0;
            elsif( (data_out_vld_s='1' and data_out_read='1') or data_out_vld_s='0') then
                must_stall <= '0';
                if(data_out_last='1') then 
                    pf_done<=data_out_last;
                else 
                    pf_done<='0';
                end if;
            else
                must_stall <= '0';
                pf_done<='0';
            end if;
        else
            must_stall <= '0';
            pf_done<='0';
        end if;
     	--      ///////// outputs /////////

         fifo_avail <= STD_LOGIC_VECTOR(TO_SIGNED(FDEPTH,CeilLog2FDepth+1) - SIGNED(fifo_rdPtr) - TO_SIGNED(1,CeilLog2FDepth+1));
     	--	// available has to be atleast one (or NumWrites) more than
     	--	// valid transactions being processed in the pipeline. Then this
     	--	// fifo buffer can accept the "valid" data in the following cycle.
     	--	// pf_ready = (fifo_avail - (num_valid_datasets * NumWrites) >=
     	--	// NumWrites);
     	--	// above expr simplified for hardware
     	--	ready_chk = (NumWrites - 1 - fifo_avail + (num_valid_datasets * NumWrites));
             ready_chk <= STD_LOGIC_VECTOR(RESIZE(((TO_SIGNED(NumWrites - 1,CeilLog2FDepth+1) - SIGNED('0' & fifo_avail)) + (SIGNED(num_valid_datasets) * TO_SIGNED(NumWrites, CeilLog2FDepth+2))), ready_chk'length));
     	if (To_integer(SIGNED(ready_chk)) < 0) then
     			pf_ready_s <= '1';
     	else
     			pf_ready_s <= '0';
     	end if;
     	pf_ready <= pf_ready_s;
     	
     -----	if (now > 0ps) then
     -----		if ( isDefined(pf_ready_s) ) then
     -----		else
     -----			--	ASSERT FALSE REPORT "pf_ready is" & pf_ready_s & 
     -----			--						"avail=" & fifo_avail & 
     -----			--						"nvd=" & num_valid_datasets & 
     -----			--						"at time " & TIME'IMAGE(NOW)
     -----			--						SEVERITY ERROR;
     -----			fail := '1';
     -----		end if;
     -----		
     -----		if ( isDefined(pf_done_s) ) then
     -----		else
     -----			--	ASSERT FALSE REPORT "pf_done is " & pf_done_s & "at time " & time'image(now) & "." SEVERITY ERROR;
     -----			fail := '1';
     -----		end if;
     -----	end if;
     ---	--  // synthesis translate_on
    end process;
    end generate pf_done_with_alldone_noiidelay ;


    
    pf_done_with_alldone : if (PfAllDoneEnable = 1) generate
   control : process (fifo_empty, data_out_read, data_in_vld, fifo_rdPtr, fifo_avail, fifo_out, pf_done_s, pf_ready_s, ready_chk, data_in_last, data_in, data_out_last, pf_continue_done, stalled, internal_done, data_out_vld_s, must_stall, read_enable, num_valid_datasets, sel_pipe   )
		variable case_select : STD_LOGIC_VECTOR(4 downto 0);
        variable case_select_for_write_en : STD_LOGIC_VECTOR(3 downto 0);
   begin 

      --	///////// internal signals /////////
      
	  sel_pipe <= fifo_empty;
  
	  if (sel_pipe = ap_const_logic_1) then
			data_out_last <= data_in_last;
			data_out_vld_s <= data_in_vld;
			data_out <= data_in;
	  else
			data_out_last <= fifo_out(DataWidth + 1);
			data_out_vld_s <= fifo_out(DataWidth);
			data_out <= fifo_out(DataWidth-1 DOWNTO 0);
	  end if;
	  data_out_vld <= data_out_vld_s and (not stalled);

	    case_select := data_out_vld_s & data_out_last & pf_continue_done & stalled & data_out_read;
        case case_select is
            when "00000" | "00001" | "00100" | "00101" => internal_done <= '0'; must_stall <= '0';
            when "00010" | "00011" | "01010" | "01011" | "10010" | "10011" | "11010" | "11011" => internal_done <= '1'; must_stall <= '1';
            when "00110" | "00111" | "01110" | "01111" | "10110" | "10111" | "11110" | "11111" => internal_done <= '1'; must_stall <= '0';
            
            when "01000" | "01001" => internal_done <= '1'; must_stall <= '1';
            when "01100" | "01101" => internal_done <= '1'; must_stall <= '0';
    
            when "11000" | "11100" => internal_done <= '0'; must_stall <= '0';
            when "11101" => internal_done <= '1'; must_stall <= '0';
            when "11001" => internal_done <= '1'; must_stall <= '1';
            when others  => internal_done <= '0'; must_stall <= '0';
		end case;

        pf_done <= internal_done and (not pf_continue_done);
 
		--      ///////// outputs /////////

        fifo_avail <= STD_LOGIC_VECTOR(TO_SIGNED(FDEPTH,CeilLog2FDepth+1) - SIGNED(fifo_rdPtr) - TO_SIGNED(1,CeilLog2FDepth+1));
		--	// available has to be atleast one (or NumWrites) more than
		--	// valid transactions being processed in the pipeline. Then this
		--	// fifo buffer can accept the "valid" data in the following cycle.
		--	// pf_ready = (fifo_avail - (num_valid_datasets * NumWrites) >=
		--	// NumWrites);
		--	// above expr simplified for hardware
		--	ready_chk = (NumWrites - 1 - fifo_avail + (num_valid_datasets * NumWrites));
        ready_chk <= STD_LOGIC_VECTOR(RESIZE(((TO_SIGNED(NumWrites - 1,CeilLog2FDepth+1) - SIGNED('0' & fifo_avail)) + (SIGNED(num_valid_datasets) * TO_SIGNED(NumWrites, CeilLog2FDepth+2))), ready_chk'length));
		if (To_integer(SIGNED(ready_chk)) < 0) then
				pf_ready_s <= '1';
		else
				pf_ready_s <= '0';
		end if;
		pf_ready <= pf_ready_s;
		
	-----	if (now > 0ps) then
	-----		if ( isDefined(pf_ready_s) ) then
	-----		else
	-----			--	ASSERT FALSE REPORT "pf_ready is" & pf_ready_s & 
	-----			--						"avail=" & fifo_avail & 
	-----			--						"nvd=" & num_valid_datasets & 
	-----			--						"at time " & TIME'IMAGE(NOW)
	-----			--						SEVERITY ERROR;
	-----			fail := '1';
	-----		end if;
	-----		
	-----		if ( isDefined(pf_done_s) ) then
	-----		else
	-----			--	ASSERT FALSE REPORT "pf_done is " & pf_done_s & "at time " & time'image(now) & "." SEVERITY ERROR;
	-----			fail := '1';
	-----		end if;
	-----	end if;
	---	--  // synthesis translate_on
   end process;
    end generate pf_done_with_alldone;

    pf_done_without_alldone : if (PfAllDoneEnable = 0) generate
      control : process (fifo_empty, data_out_read, data_in_vld, fifo_rdPtr, fifo_avail, fifo_out, pf_done_s, pf_ready_s, ready_chk, data_in_last, data_in, data_out_last, pf_continue_done, stalled, internal_done, data_out_vld_s, must_stall, read_enable, num_valid_datasets, sel_pipe   )
       	variable case_select : STD_LOGIC_VECTOR(4 downto 0);
           variable case_select_for_write_en : STD_LOGIC_VECTOR(3 downto 0);
      begin 

         --	///////// internal signals /////////
         
         sel_pipe <= fifo_empty;
  
         if (sel_pipe = ap_const_logic_1) then
       		data_out_last <= data_in_last;
       		data_out_vld_s <= data_in_vld;
       		data_out <= data_in;
         else
       		data_out_last <= fifo_out(DataWidth + 1);
       		data_out_vld_s <= fifo_out(DataWidth);
       		data_out <= fifo_out(DataWidth-1 DOWNTO 0);
         end if;
         data_out_vld <= data_out_vld_s and (not stalled);

           case_select := data_out_vld_s & data_out_last & pf_continue_done & stalled & data_out_read;
           case case_select is
               when "00000" | "00001" | "00100" | "00101" => internal_done <= '0'; must_stall <= '0';
               when "00010" | "00011" | "01010" | "01011" | "10010" | "10011" | "11010" | "11011" => internal_done <= '1'; must_stall <= '1';
               when "00110" | "00111" | "01110" | "01111" | "10110" | "10111" | "11110" | "11111" => internal_done <= '1'; must_stall <= '0';
               
               when "01000" | "01001" => internal_done <= '1'; must_stall <= '1';
               when "01100" | "01101" => internal_done <= '1'; must_stall <= '0';
       
               when "11000" | "11100" => internal_done <= '0'; must_stall <= '0';
               when "11101" => internal_done <= '1'; must_stall <= '0';
               when "11001" => internal_done <= '1'; must_stall <= '1';
               when others  => internal_done <= '0'; must_stall <= '0';
       	end case;

               pf_done <= internal_done;
 
       	--      ///////// outputs /////////

           fifo_avail <= STD_LOGIC_VECTOR(TO_SIGNED(FDEPTH,CeilLog2FDepth+1) - SIGNED(fifo_rdPtr) - TO_SIGNED(1,CeilLog2FDepth+1));
       	--	// available has to be atleast one (or NumWrites) more than
       	--	// valid transactions being processed in the pipeline. Then this
       	--	// fifo buffer can accept the "valid" data in the following cycle.
       	--	// pf_ready = (fifo_avail - (num_valid_datasets * NumWrites) >=
       	--	// NumWrites);
       	--	// above expr simplified for hardware
       	--	ready_chk = (NumWrites - 1 - fifo_avail + (num_valid_datasets * NumWrites));
           ready_chk <= STD_LOGIC_VECTOR(RESIZE(((TO_SIGNED(NumWrites - 1,CeilLog2FDepth+1) - SIGNED('0' & fifo_avail)) + (SIGNED(num_valid_datasets) * TO_SIGNED(NumWrites, CeilLog2FDepth+2))), ready_chk'length));
       	if (To_integer(SIGNED(ready_chk)) < 0) then
       			pf_ready_s <= '1';
       	else
       			pf_ready_s <= '0';
       	end if;
       	pf_ready <= pf_ready_s;
       	
       -----	if (now > 0ps) then
       -----		if ( isDefined(pf_ready_s) ) then
       -----		else
       -----			--	ASSERT FALSE REPORT "pf_ready is" & pf_ready_s & 
       -----			--						"avail=" & fifo_avail & 
       -----			--						"nvd=" & num_valid_datasets & 
       -----			--						"at time " & TIME'IMAGE(NOW)
       -----			--						SEVERITY ERROR;
       -----			fail := '1';
       -----		end if;
       -----		
       -----		if ( isDefined(pf_done_s) ) then
       -----		else
       -----			--	ASSERT FALSE REPORT "pf_done is " & pf_done_s & "at time " & time'image(now) & "." SEVERITY ERROR;
       -----			fail := '1';
       -----		end if;
       -----	end if;
       ---	--  // synthesis translate_on
      end process;
    end generate pf_done_without_alldone;

--   process (ap_clk)
--   begin
--		if ap_clk'event and ap_clk = '1' then
--			if ( isDefined(data_out_read) ) then
--			else
--				--	ASSERT FALSE REPORT "data_out_read is " & data_out_read & "." SEVERITY ERROR;
--				fail := '1';
--			end if;
--		end if;
--   end process;
   

--   process
--   begin
--      if (fail = ap_const_logic_1) then
--		ASSERT FALSE
--		REPORT "<<<<<<<< Abort! Assertion failure at time " & time'image(now) & "  >>>>>>>>"
--		SEVERITY FAILURE;
--		wait until rising_edge(ap_clk);
--		wait until rising_edge(ap_clk);
--		wait;
--      end if;
--   end process;
   -----	// synthesis translate_on   
 
end behav;
    
