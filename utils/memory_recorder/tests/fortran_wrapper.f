      program memory_recorder_fortran_test

        use memory_recorder

        use mpi

        ! Translate main.cpp to fortran.

        integer :: rank, sz, ierror, mb, array_size, k, argc
        real, allocatable, dimension(:) :: s1, s2, s3
        character(len=32) :: argin

        argc = command_argument_count()
        mb = 1024*1024
        call MPI_INIT(ierror)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, sz, ierror)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

        array_size = 64
        if (argc > 0) then
          call get_command_argument(1, argin)
          read (argin,'(I32)') array_size
        end if

        array_size = mb*array_size
        call start_memrecorder()

        allocate(s1(array_size), s2(array_size), s3(array_size))

        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
        
        call read_meminfo("PostMalloc")
        do k = 1, array_size
          s1(k) = real(time())
          s2(k) = real(time())
          s3(k) = s2(k) + s1(k)
        end do
        
        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
        call read_meminfo("PostFill")
        
        deallocate(s1, s2, s3)
        
        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
        call read_meminfo("PostFree")
        
        call write_meminfo()
        call write_rss()
        
        call free_memrecorder()
        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
        call MPI_FINALIZE(ierror)

      end program memory_recorder_fortran_test