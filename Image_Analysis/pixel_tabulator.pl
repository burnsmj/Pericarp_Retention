#! /usr/bin/perl
### Pixel Color Tabulation
### Candy Hirsch and Michael J Burns
### 4/16/2021

####################
# Starting Options #
####################
use strict;
use warnings;
use Getopt::Std;

#####################
# Data Requirements #
#####################
# Data must be in the format described by plant cv.  See below for an example.
# #pericarp
# 111,123,223 97,96,54  43,67,89
# #endosperm
# 23,34,45  12,23,34  67,78,89
# #background
# 1,1,1 2,2,2 3,3,3
# The data must be in the order of pericarp, endosperm, background
# There cannot be blank rows between sections
# There cannot be blank spaces between the # and the tissue descriptor
# The tissue descriptor must be lower case

####################
# Usage Statements #
####################
my $usage = "\n$0 -i <INPUT DATA> -o <OUTPUT DATA> -h <help>\n\n"; # statement to print if something goes wrong
our ($opt_i, $opt_o, $opt_h); # defining options
getopts("i:o:h") or die "\nGetopts Problem: $usage\n"; # requiring our options

if((!(defined $opt_i)) or (!(defined $opt_o)) or (defined $opt_h)) { # checking to make sure all needed options are defined
  die "\n $usage"; # stop and print usage statement if something goes wrong
}

# Example Usage: perl Pixel_Tabulator.pl -i white_corn_pixel_data.txt -o white_corn_pixel_tabulated_data.txt

###################
# Open File Paths #
###################
open (my $in_fh, '<', $opt_i) || die;
open (my $out_fh, '>', $opt_o) || die;

######################
# Initiate Variables #
######################
my %tissues;
my $tissue;
my $flag = 0;

###################
# Populating Hash #
###################
while (my $line = <$in_fh>) {
  chomp $line;
  if ($line =~ /^#(\S+)/) {
    $tissue = $1;
    $flag = 1;
  }

  else {
    if ($flag == 0) {
      $tissues{$tissue} .= "\t$line";
    }
    elsif ($flag == 1) {
      $tissues{$tissue} = $line;
      $flag = 0;
    }
  }
}

##########################
# Defining Column Arrays #
##########################
my @endosperm = split ('\t', $tissues{'endosperm'});
my @pericarp = split ('\t', $tissues{'pericarp'});
my @background = split ('\t', $tissues{'background'});

my $count = @endosperm;

###################
# Tabulating Data #
###################
print $out_fh "endosperm\tpericarp\tbackground\n";
my $i;
for ($i = 0; $i < $count; ++$i) {
  print $out_fh "$endosperm[$i]\t$pericarp[$i]\t$background[$i]\n";
}

######################
# Closing File Paths #
######################
close $in_fh;
close $out_fh;
